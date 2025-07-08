use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use chrono::{DateTime, FixedOffset, Utc};
use parking_lot::Mutex;
use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::types::{IntoPyDict, PyDict};
use pyo3::PyAny;
use pyo3::{prelude::*, IntoPyObjectExt};
use ratatui::style::Color;
use rusqlite::Connection;
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum TaskStatus {
    Running = 0,
    Failed = 1,
    Queued = 2,
    Pending = 3,
    Done = 4,
    Skipped = 5,
}
impl TaskStatus {
    fn color(&self) -> Color {
        match self {
            TaskStatus::Running => catppuccin::PALETTE.mocha.colors.blue.into(),
            TaskStatus::Failed => catppuccin::PALETTE.mocha.colors.red.into(),
            TaskStatus::Queued => catppuccin::PALETTE.mocha.colors.mauve.into(),
            TaskStatus::Pending => catppuccin::PALETTE.mocha.colors.peach.into(),
            TaskStatus::Done => catppuccin::PALETTE.mocha.colors.green.into(),
            TaskStatus::Skipped => catppuccin::PALETTE.mocha.colors.yellow.into(),
        }
    }
}
impl Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TaskStatus::Running => "running",
                TaskStatus::Failed => "failed",
                TaskStatus::Queued => "queued",
                TaskStatus::Pending => "pending",
                TaskStatus::Done => "done",
                TaskStatus::Skipped => "skipped",
            }
        )
    }
}
impl FromStr for TaskStatus {
    type Err = PyErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "running" => Ok(TaskStatus::Running),
            "failed" => Ok(TaskStatus::Failed),
            "queued" => Ok(TaskStatus::Queued),
            "pending" => Ok(TaskStatus::Pending),
            "done" => Ok(TaskStatus::Done),
            "skipped" => Ok(TaskStatus::Skipped),
            _ => Err(PyValueError::new_err("Invalid task status")),
        }
    }
}

#[derive(Clone, Debug)]
struct TaskRecord {
    name: String,
    status: TaskStatus,
    inputs: Vec<String>,
    outputs: Vec<PathBuf>,
    resources: HashMap<String, usize>,
    isolated: bool,
    log_path: PathBuf,
    start_time: DateTime<FixedOffset>,
    end_time: DateTime<FixedOffset>,
    payload: String,
}
impl TaskRecord {
    fn clone_with_status(&self, status: TaskStatus) -> Self {
        let mut out = self.clone();
        out.status = status;
        out
    }
    fn clone_with_start_time(&self, start_time: DateTime<FixedOffset>) -> Self {
        let mut out = self.clone();
        out.start_time = start_time;
        out
    }
    fn clone_with_end_time(&self, end_time: DateTime<FixedOffset>) -> Self {
        let mut out = self.clone();
        out.end_time = end_time;
        out
    }
    fn as_pydict<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let key_vals: &[(&str, PyObject)] = &[
            ("name", self.name.clone().into_py_any(py)?),
            ("status", self.status.to_string().into_py_any(py)?),
            ("inputs", self.inputs.clone().into_py_any(py)?),
            ("outputs", self.outputs.clone().into_py_any(py)?),
            ("resources", self.resources.clone().into_py_any(py)?),
            ("isolated", self.isolated.into_py_any(py)?),
            ("log_path", self.log_path.clone().into_py_any(py)?),
            ("start_time", self.start_time.to_string().into_py_any(py)?),
            ("end_time", self.end_time.to_string().into_py_any(py)?),
            ("payload", self.payload.clone().into_py_any(py)?),
        ];
        key_vals.into_py_dict(py)
    }
}
struct Database {
    conn: Arc<Mutex<Connection>>,
}
impl Database {
    pub fn new(project: &str, base_path: Option<PathBuf>) -> PyResult<Self> {
        let base_dir = match base_path {
            Some(p) => p,
            None => {
                let mut default_path = dirs::home_dir()
                    .ok_or_else(|| PyIOError::new_err("Could not determine home directory"))?;
                default_path.push(".modak");
                default_path
            }
        };

        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir).map_err(|e| {
                PyIOError::new_err(format!("Failed to create .modak directory: {e}"))
            })?;
        }

        let mut project_db_path = base_dir.clone();
        project_db_path.push(format!("{}.db", project));

        let create_schema = !project_db_path.exists();

        let conn = Connection::open(&project_db_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open database: {e}")))?;
        conn.pragma_update(None, "journal_mode", &"WAL")
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        conn.busy_timeout(Duration::from_secs(30))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        if create_schema {
            conn.execute_batch(
                "
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY,
                    job_name TEXT NOT NULL UNIQUE,
                    inputs TEXT NOT NULL,
                    outputs TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    resources TEXT NOT NULL,
                    isolated INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                ",
            )
            .map_err(|e| PyIOError::new_err(format!("Failed to initialize schema: {e}")))?;
        }

        Ok(Database {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
    fn upsert_task(&self, task: &TaskRecord) -> PyResult<()> {
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        tx.execute(
            "
            INSERT INTO jobs (
                job_name, inputs, outputs, log_path, status,
                start_time, end_time, resources, isolated, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_name)
            DO UPDATE SET
                inputs = excluded.inputs,
                outputs = excluded.outputs,
                log_path = excluded.log_path,
                status = excluded.status,
                start_time = excluded.start_time,
                end_time = excluded.end_time,
                resources = excluded.resources,
                isolated = excluded.isolated,
                payload = excluded.payload
            ",
            rusqlite::params![
                task.name,
                serde_json::to_string(&task.inputs).unwrap(),
                serde_json::to_string(&task.outputs).unwrap(),
                task.log_path.to_string_lossy(),
                task.status.to_string(),
                task.start_time.to_rfc3339(),
                task.end_time.to_rfc3339(),
                serde_json::to_string(&task.resources).unwrap(),
                task.isolated as i32,
                task.payload,
            ],
        )
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

        tx.commit().map_err(|e| PyIOError::new_err(e.to_string()))
    }
    fn batch_upsert_task(&self, tasks: &[TaskRecord]) -> PyResult<()> {
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        {
            let mut stmt = tx
                .prepare(
                    "
            INSERT INTO jobs (
                job_name, inputs, outputs, log_path, status,
                start_time, end_time, resources, isolated, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_name)
            DO UPDATE SET
                inputs = excluded.inputs,
                outputs = excluded.outputs,
                log_path = excluded.log_path,
                status = excluded.status,
                start_time = excluded.start_time,
                end_time = excluded.end_time,
                resources = excluded.resources,
                isolated = excluded.isolated,
                payload = excluded.payload
            ",
                )
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            for task in tasks {
                stmt.execute(rusqlite::params![
                    task.name,
                    serde_json::to_string(&task.inputs).unwrap(),
                    serde_json::to_string(&task.outputs).unwrap(),
                    task.log_path.to_string_lossy(),
                    task.status.to_string(),
                    task.start_time.to_rfc3339(),
                    task.end_time.to_rfc3339(),
                    serde_json::to_string(&task.resources).unwrap(),
                    task.isolated as i32,
                    task.payload,
                ])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
        }
        tx.commit().map_err(|e| PyIOError::new_err(e.to_string()))
    }
    fn get_project_state(&self) -> PyResult<Vec<TaskRecord>> {
        let conn = self.conn.lock();

        let mut stmt = conn
            .prepare(
                "
            SELECT job_name, inputs, outputs, log_path, status,
                   start_time, end_time, resources, isolated, payload
            FROM jobs
            ",
            )
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(TaskRecord {
                    name: row.get(0)?,
                    inputs: serde_json::from_str(&row.get::<_, String>(1)?).unwrap(),
                    outputs: serde_json::from_str(&row.get::<_, String>(2)?).unwrap(),
                    log_path: PathBuf::from(row.get::<_, String>(3)?),
                    status: row.get::<_, String>(4)?.parse().unwrap(),
                    start_time: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?).unwrap(),
                    end_time: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?).unwrap(),
                    resources: serde_json::from_str(&row.get::<_, String>(7)?).unwrap(),
                    isolated: row.get::<_, i32>(8)? != 0,
                    payload: row.get(9)?,
                })
            })
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let state = rows
            .collect::<Result<Vec<TaskRecord>, _>>()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(state)
    }
    fn reset_project(&self) -> PyResult<()> {
        let conn = self.conn.lock();

        conn.execute("DELETE FROM jobs", [])
            .map_err(|e| PyIOError::new_err(format!("Failed to clear jobs table: {e}")))?;

        Ok(())
    }
    fn get_input_tasks(&self, _project: &str, task_name: &str) -> PyResult<Vec<TaskRecord>> {
        let conn = self.conn.lock();

        let inputs_json: String = conn
            .query_row(
                "SELECT inputs FROM jobs WHERE job_name = ?",
                [task_name],
                |row| row.get(0),
            )
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let input_names: Vec<String> =
            serde_json::from_str(&inputs_json).map_err(|e| PyIOError::new_err(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT job_name, inputs, outputs, log_path, status, start_time, end_time, resources, isolated, payload
             FROM jobs WHERE job_name = ?",
        ).map_err(|e| PyIOError::new_err(e.to_string()))?;

        let mut results = Vec::new();
        for input in input_names {
            let mut rows = stmt
                .query_map(rusqlite::params![input], |row| {
                    Ok(TaskRecord {
                        name: row.get(0)?,
                        inputs: serde_json::from_str(&row.get::<_, String>(1)?).unwrap(),
                        outputs: serde_json::from_str(&row.get::<_, String>(2)?).unwrap(),
                        log_path: PathBuf::from(row.get::<_, String>(3)?),
                        status: row.get::<_, String>(4)?.parse().unwrap(),
                        start_time: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                            .unwrap(),
                        end_time: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?).unwrap(),
                        resources: serde_json::from_str(&row.get::<_, String>(7)?).unwrap(),
                        isolated: row.get::<_, i32>(8)? != 0,
                        payload: row.get(9)?,
                    })
                })
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            if let Some(row) = rows.next() {
                results.push(row.map_err(|e| PyIOError::new_err(e.to_string()))?)
            }
        }
        Ok(results)
    }
    fn list_projects(base_path: Option<PathBuf>) -> PyResult<Vec<String>> {
        let base_dir = match base_path {
            Some(p) => p,
            None => {
                let mut default_path = dirs::home_dir()
                    .ok_or_else(|| PyIOError::new_err("Could not determine home directory"))?;
                default_path.push(".modak");
                default_path
            }
        };

        if !base_dir.exists() {
            return Ok(Vec::new()); // No directory, so no projects
        }

        let mut projects = Vec::new();
        for entry in std::fs::read_dir(&base_dir)
            .map_err(|e| PyIOError::new_err(format!("Failed to read .modak directory: {e}")))?
        {
            let entry = entry.map_err(|e| PyIOError::new_err(e.to_string()))?;
            let path = entry.path();
            if path.is_file() {
                if let Some(os_str_name) = path.file_name() {
                    let file_name = os_str_name.to_string_lossy();
                    if file_name.ends_with(".db") {
                        let project_name = file_name.trim_end_matches(".db").to_string();
                        projects.push(project_name);
                    }
                }
            }
        }
        projects.sort();
        Ok(projects)
    }
}

/// A queue for Tasks.
///
/// Arguments
/// ---------
/// project : str
///     The name of the project.
/// workers : int, default=4
///     The maximum number of tasks which can run in parallel.
/// resources : dict of str to int, optional
///     The available resources for the entire queue.
/// log_path : Path, optional
///     If provided, this file will act as a global log for all tasks.
/// state_file_path : Path, optional
///     The location of the state file used to track tasks. This defaults
///     to the $HOME/.modak/state.db
///
/// Returns
/// -------
/// TaskQueue
///
#[pyclass]
pub struct TaskQueue {
    project: String,
    max_workers: usize,
    available_resources: HashMap<String, usize>,
    running: HashMap<String, std::thread::JoinHandle<i32>>,
    log_file_path: Option<PathBuf>,
    database: Database,
}

#[pymethods]
impl TaskQueue {
    #[new]
    #[pyo3(signature = (project, *, workers = 4, resources = None, log_path = None, state_file_path = None))]
    fn new(
        project: String,
        workers: usize,
        resources: Option<HashMap<String, usize>>,
        log_path: Option<PathBuf>,
        state_file_path: Option<PathBuf>,
    ) -> PyResult<Self> {
        Ok(TaskQueue {
            project: project.clone(),
            max_workers: workers,
            available_resources: resources.unwrap_or_default(),
            running: HashMap::new(),
            log_file_path: log_path,
            database: Database::new(&project, state_file_path)?,
        })
    }

    /// Run a set of Tasks in parallel.
    ///
    /// Arguments
    /// ---------
    /// tasks: list of Task
    ///     The tasks to run in parallel. Note that this only needs to include tasks which are at the
    ///     end of a pipeline, as dependencies are discovered automatically, but duplicate tasks will
    ///     not be run multiple times if included.
    ///
    /// Returns
    /// -------
    /// None
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a cycle is detected in the graph of tasks or a dependency chain is corrupted in some
    ///     way.
    /// IOError
    ///     If the state file cannot be written to or read from
    ///
    fn run(&mut self, tasks: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        // clear out previous runs
        self.database.reset_project()?;
        let mut task_objs = vec![];
        let mut seen = HashSet::new();
        let mut stack = tasks;

        // get rid of duplicates and traverse graph to add inputs to list of task_objs
        while let Some(obj) = stack.pop() {
            let task_name = obj.getattr("name")?.extract::<String>()?;
            if seen.contains(&task_name) {
                continue;
            }
            seen.insert(task_name);
            stack.extend(obj.getattr("inputs")?.extract::<Vec<Bound<'_, PyAny>>>()?);
            task_objs.push(obj);
        }

        // create a mapping from task name to index in task_objs for convenience
        let mut name_to_index = HashMap::new();
        for (i, obj) in task_objs.iter().enumerate() {
            name_to_index.insert(obj.getattr("name")?.extract::<String>()?, i);
        }

        let mut graph: DiGraphMap<usize, ()> = DiGraphMap::new();
        for (i, obj) in task_objs.iter().enumerate() {
            graph.add_node(i);
            let inputs: Vec<Bound<'_, PyAny>> = obj.getattr("inputs")?.extract()?;
            for inp in inputs {
                if let Some(&src) = name_to_index.get(&inp.getattr("name")?.extract::<String>()?) {
                    graph.add_edge(src, i, ());
                }
            }
        }

        let sorted = toposort(&graph, None)
            .map_err(|_| PyErr::new::<PyValueError, _>("Cycle in task graph"))?;

        let mut tasks = Vec::with_capacity(sorted.len());
        for id in sorted {
            let task_obj = &task_objs[id];
            let name: String = task_obj.getattr("name")?.extract()?;
            let py_inputs: Vec<Bound<'_, PyAny>> = task_obj.getattr("inputs")?.extract()?;
            let mut inputs = Vec::new();
            for py_obj in py_inputs {
                let input_name: String = py_obj.getattr("name")?.extract()?;
                if name_to_index.contains_key(&input_name) {
                    inputs.push(input_name);
                } else {
                    return Err(PyErr::new::<PyValueError, _>(
                        "Unrecognized input task object",
                    ));
                }
            }

            let outputs: Vec<PathBuf> = task_obj.getattr("outputs")?.extract()?;
            let resources: HashMap<String, usize> = task_obj.getattr("resources")?.extract()?;
            let isolated: bool = task_obj.getattr("isolated")?.extract()?;
            let payload: String = task_obj.call_method0("serialize")?.extract()?;
            let log_path: PathBuf = task_obj.getattr("log_path")?.extract()?;
            let mut status = TaskStatus::Pending;
            if !outputs.is_empty() && outputs.iter().all(|p| p.exists()) {
                status = TaskStatus::Skipped;
            }
            if self
                .available_resources
                .iter()
                .any(|(resource_name, amount)| resources.get(resource_name).unwrap_or(&0) > amount)
            {
                status = TaskStatus::Failed;
            }
            let log_behavior: String = task_obj.getattr("log_behavior")?.extract()?;
            if log_behavior == "overwrite" && log_path.exists() {
                std::fs::remove_file(&log_path).map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
            let start_time = DateTime::default();
            let end_time = DateTime::default();
            let record = TaskRecord {
                name,
                status,
                inputs,
                outputs,
                resources,
                isolated,
                log_path,
                start_time,
                end_time,
                payload,
            };
            tasks.push(record);
        }
        self.database.batch_upsert_task(&tasks)?;
        // skip container jobs with no outputs where all inputs are complete
        for task in self.database.get_project_state()? {
            if task.outputs.is_empty() && self.can_queue(&task)? {
                self.database
                    .upsert_task(&task.clone_with_status(TaskStatus::Skipped))?;
            }
        }
        // Now that we have filled the database with the current state, we can run the tasks
        loop {
            thread::sleep(Duration::from_millis(50));
            // First, we get the current state of all tasks in the project:
            let tasks = self.database.get_project_state()?;
            // If all done, skipped, or failed, stop looping
            if tasks.iter().all(|t| {
                matches!(
                    t.status,
                    TaskStatus::Done | TaskStatus::Skipped | TaskStatus::Failed
                )
            }) {
                break;
            }
            for task in &tasks {
                match task.status {
                    TaskStatus::Pending => {
                        if self.can_queue(&task)? {
                            self.database
                                .upsert_task(&task.clone_with_status(TaskStatus::Queued))?;
                        } else {
                            continue;
                        }
                    }
                    TaskStatus::Queued => {
                        if self.can_run(&task) {
                            for (resource, amount) in self.available_resources.iter_mut() {
                                if let Some(req_amount) = task.resources.get(resource) {
                                    *amount -= req_amount;
                                }
                            }
                            let payload = task.payload.clone();
                            let handle = if let Some(log_path) = self.log_file_path.clone() {
                                thread::spawn(move || {
                                    let mut temp_file =
                                        NamedTempFile::new().expect("Failed to create temp file");
                                    temp_file
                                        .write_all(payload.as_bytes())
                                        .expect("Failed to write payload to temp file");
                                    let path = temp_file.path().to_owned();
                                    let status = Command::new("python3")
                                        .arg("-m")
                                        .arg("modak")
                                        .arg(path)
                                        .arg(log_path)
                                        .status()
                                        .unwrap();
                                    drop(temp_file);
                                    status.code().unwrap()
                                })
                            } else {
                                thread::spawn(move || {
                                    let mut temp_file =
                                        NamedTempFile::new().expect("Failed to create temp file");
                                    temp_file
                                        .write_all(payload.as_bytes())
                                        .expect("Failed to write payload to temp file");
                                    let path = temp_file.path().to_owned();
                                    let status = Command::new("python3")
                                        .arg("-m")
                                        .arg("modak")
                                        .arg(path)
                                        .status()
                                        .unwrap();
                                    drop(temp_file);
                                    status.code().unwrap()
                                })
                            };
                            self.running.insert(task.name.clone(), handle);
                            self.database.upsert_task(
                                &task
                                    .clone_with_status(TaskStatus::Running)
                                    .clone_with_start_time(Utc::now().into()),
                            )?;
                        }
                    }
                    TaskStatus::Running => {
                        let handle = self.running.remove(&task.name).unwrap();
                        // if task is finished, update the database and return resources
                        if handle.is_finished() {
                            match handle.join() {
                                Ok(status) => match status {
                                    0 => {
                                        self.database.upsert_task(
                                            &task
                                                .clone_with_status(TaskStatus::Done)
                                                .clone_with_end_time(Utc::now().into()),
                                        )?;
                                    }
                                    _ => {
                                        self.database.upsert_task(
                                            &task
                                                .clone_with_status(TaskStatus::Failed)
                                                .clone_with_end_time(Utc::now().into()),
                                        )?;
                                    }
                                },
                                Err(e) => {
                                    eprintln!("Task {} failed: {:?}", task.name, e);
                                    self.database.upsert_task(
                                        &task
                                            .clone_with_status(TaskStatus::Failed)
                                            .clone_with_end_time(Utc::now().into()),
                                    )?;
                                }
                            }
                            for (resource, amount) in self.available_resources.iter_mut() {
                                if let Some(req_amount) = task.resources.get(resource) {
                                    *amount += req_amount;
                                }
                            }
                        } else {
                            // if the task isn't finished, reinsert it into the running list
                            self.running.insert(task.name.clone(), handle);
                        }
                    }
                    TaskStatus::Failed => {
                        // if the task failed, go through all other tasks, check if the failed task
                        // is in their input lists, and if so, fail that task too
                        for d_task in &tasks {
                            let d_task_input_names: Vec<String> = self
                                .database
                                .get_input_tasks(&self.project, &d_task.name)?
                                .iter()
                                .map(|t| t.name.clone())
                                .collect();
                            if d_task_input_names.contains(&task.name) {
                                self.database
                                    .upsert_task(&task.clone_with_status(TaskStatus::Failed))?;
                            }
                        }
                    }
                    TaskStatus::Done | TaskStatus::Skipped => continue,
                }
            }
        }
        Ok(())
    }
}

impl TaskQueue {
    fn can_queue(&self, task: &TaskRecord) -> PyResult<bool> {
        let input_tasks = self.database.get_input_tasks(&self.project, &task.name)?;
        for input_task in input_tasks {
            if matches!(input_task.status, TaskStatus::Done | TaskStatus::Skipped) {
                for output_path_str in &input_task.outputs {
                    let path = Path::new(&output_path_str);
                    if !path.exists() {
                        return Ok(false);
                    }
                }
            } else {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn can_run(&self, task: &TaskRecord) -> bool {
        (!task.isolated || self.running.is_empty())
            && self
                .available_resources
                .iter()
                .all(|(resource_name, available_amount)| {
                    task.resources.get(resource_name).unwrap_or(&0) <= available_amount
                })
            && self.max_workers > self.running.len()
    }
}

mod queue_app;
use queue_app::QueueApp;

#[pyfunction]
fn run_queue_wrapper(state_file_path: Option<PathBuf>, project: Option<String>) -> PyResult<()> {
    let mut terminal = ratatui::init();
    let result = QueueApp::new(state_file_path, project)?.run(&mut terminal);
    ratatui::restore();
    result.map_err(|e| PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn get_projects(state_file_path: Option<PathBuf>) -> PyResult<Vec<String>> {
    Database::list_projects(state_file_path)
}

#[pyfunction]
fn get_project_state(
    py: Python,
    state_file_path: Option<PathBuf>,
    project: String,
) -> PyResult<Vec<Bound<PyDict>>> {
    let database = Database::new(&project, state_file_path)?;
    database
        .get_project_state()?
        .into_iter()
        .map(|state| state.as_pydict(py))
        .collect::<PyResult<Vec<Bound<PyDict>>>>()
}

#[pyfunction]
fn reset_project(state_file_path: Option<PathBuf>, project: String) -> PyResult<()> {
    let database = Database::new(&project, state_file_path)?;
    database.reset_project()
}

#[pymodule]
fn modak(m: Bound<PyModule>) -> PyResult<()> {
    m.add_class::<TaskQueue>()?;
    m.add_function(wrap_pyfunction!(run_queue_wrapper, &m)?)?;
    m.add_function(wrap_pyfunction!(get_projects, &m)?)?;
    m.add_function(wrap_pyfunction!(get_project_state, &m)?)?;
    m.add_function(wrap_pyfunction!(reset_project, &m)?)?;
    Ok(())
}
