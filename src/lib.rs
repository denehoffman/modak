use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Display;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use chrono::Utc;
use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::PyAny;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

#[derive(Debug, Clone, PartialEq, Eq)]
enum TaskStatus {
    Pending,
    Queued,
    Running,
    Done,
    Failed,
    Skipped,
}
impl Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "pending"),
            TaskStatus::Queued => write!(f, "queued"),
            TaskStatus::Running => write!(f, "running"),
            TaskStatus::Done => write!(f, "done"),
            TaskStatus::Failed => write!(f, "failed"),
            TaskStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug)]
struct TaskMeta {
    name: String,
    inputs: Vec<usize>,
    outputs: Vec<PathBuf>,
    resources: HashMap<String, usize>,
    isolated: bool,
    payload: String,
    log_path: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct TaskState {
    status: String,
    inputs: Vec<String>,
    outputs: Vec<PathBuf>,
    resources: HashMap<String, usize>,
    isolated: bool,
    log_path: PathBuf,
    start_time: String,
    end_time: String,
}

type ModakState = BTreeMap<String, TaskState>;

/// A queue for Tasks.
///
/// Arguments
/// ---------
/// workers : int, default=4
///     The maximum number of tasks which can run in parallel.
/// resources : dict of str to int, optional
///     The available resources for the entire queue.
/// state_file_path : Path, default=".modak"
///     The location of the state file used to track tasks.
/// log_path : Path, optional
///     If provided, this file will act as a global log for all tasks.
///
/// Returns
/// -------
/// TaskQueue
///
#[pyclass]
pub struct TaskQueue {
    tasks: HashMap<usize, TaskMeta>,
    statuses: HashMap<usize, TaskStatus>,
    timestamps: HashMap<usize, (String, String)>,
    max_workers: usize,
    available_resources: HashMap<String, usize>,
    running: HashMap<usize, std::thread::JoinHandle<i32>>,
    state_file_path: PathBuf,
    log_file_path: Option<PathBuf>,
}

#[pymethods]
impl TaskQueue {
    #[new]
    #[pyo3(signature = (*, workers = 4, resources = None, state_file_path = None, log_path = None))]
    fn new(
        workers: usize,
        resources: Option<HashMap<String, usize>>,
        state_file_path: Option<PathBuf>,
        log_path: Option<PathBuf>,
    ) -> Self {
        TaskQueue {
            tasks: HashMap::new(),
            statuses: HashMap::new(),
            timestamps: HashMap::new(),
            max_workers: workers,
            available_resources: resources.unwrap_or_default(),
            running: HashMap::new(),
            state_file_path: state_file_path.unwrap_or(PathBuf::from(".modak")),
            log_file_path: log_path,
        }
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
        let mut task_objs = vec![];
        let mut seen = HashSet::new();
        let mut stack = tasks;

        while let Some(obj) = stack.pop() {
            let task_hash = obj.hash()?;
            if seen.contains(&task_hash) {
                continue;
            }
            seen.insert(task_hash);
            stack.extend(obj.getattr("inputs")?.extract::<Vec<Bound<'_, PyAny>>>()?);
            task_objs.push(obj);
        }

        let mut obj_to_index = HashMap::new();
        for (i, obj) in task_objs.iter().enumerate() {
            obj_to_index.insert(obj.hash()?, i);
        }

        let mut graph: DiGraphMap<usize, ()> = DiGraphMap::new();
        for (i, obj) in task_objs.iter().enumerate() {
            graph.add_node(i);
            let inputs: Vec<Bound<'_, PyAny>> = obj.getattr("inputs")?.extract()?;
            for inp in inputs {
                if let Some(&src) = obj_to_index.get(&inp.hash()?) {
                    graph.add_edge(src, i, ());
                }
            }
        }

        let sorted = toposort(&graph, None)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Cycle in task graph"))?;

        for id in sorted {
            let task_obj = &task_objs[id];
            let py_inputs: Vec<Bound<'_, PyAny>> = task_obj.getattr("inputs")?.extract()?;
            let mut inputs = Vec::new();
            for py_obj in py_inputs {
                match obj_to_index.get(&py_obj.hash()?) {
                    Some(&idx) => inputs.push(idx),
                    None => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Unrecognized input task object",
                        ))
                    }
                }
            }

            let name: String = task_obj.getattr("name")?.extract()?;
            let outputs: Vec<PathBuf> = task_obj.getattr("outputs")?.extract()?;
            let resources: HashMap<String, usize> = task_obj.getattr("resources")?.extract()?;
            let isolated: bool = task_obj.getattr("isolated")?.extract()?;
            let payload: String = task_obj.call_method0("serialize")?.extract()?;
            let log_path: PathBuf = task_obj.getattr("log_path")?.extract()?;

            if !outputs.is_empty() && outputs.iter().all(|p| p.exists()) {
                self.statuses.insert(id, TaskStatus::Skipped);
            } else {
                self.statuses.insert(id, TaskStatus::Pending);
            }
            if self
                .available_resources
                .iter()
                .any(|(resoruce_name, amount)| resources.get(resoruce_name).unwrap_or(&0) > amount)
            {
                self.statuses.insert(id, TaskStatus::Failed);
            }
            self.timestamps.insert(id, ("".to_string(), "".to_string()));

            self.tasks.insert(
                id,
                TaskMeta {
                    name,
                    inputs,
                    outputs,
                    resources,
                    isolated,
                    payload,
                    log_path,
                },
            );
        }
        self.update_state_file()?;
        loop {
            thread::sleep(Duration::from_millis(50));
            if self.all_done() {
                break;
            }
            for (id, task) in self.tasks.iter() {
                match self.statuses.get(id).unwrap() {
                    TaskStatus::Pending => {
                        if self.can_queue(task) {
                            self.statuses.insert(*id, TaskStatus::Queued);
                        } else {
                            continue;
                        }
                    }
                    TaskStatus::Queued => {
                        if self.can_run(task) {
                            self.statuses.insert(*id, TaskStatus::Running);
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
                            self.running.insert(*id, handle);
                            self.timestamps
                                .insert(*id, (Utc::now().to_rfc3339(), "".to_string()));
                        }
                    }
                    TaskStatus::Running => {
                        let handle = self.running.remove(id).unwrap();
                        if handle.is_finished() {
                            match handle.join() {
                                Ok(status) => match status {
                                    0 => {
                                        self.statuses.insert(*id, TaskStatus::Done);
                                    }
                                    _ => {
                                        self.statuses.insert(*id, TaskStatus::Failed);
                                    }
                                },
                                Err(e) => {
                                    eprintln!("Task {id} failed: {e:?}");
                                    self.statuses.insert(*id, TaskStatus::Failed);
                                }
                            }
                            for (resource, amount) in self.available_resources.iter_mut() {
                                if let Some(req_amount) = task.resources.get(resource) {
                                    *amount += req_amount;
                                }
                            }
                            self.running.remove(id);
                            let (start, _) = self.timestamps.get(id).unwrap();
                            self.timestamps
                                .insert(*id, (start.to_string(), Utc::now().to_rfc3339()));
                        } else {
                            self.running.insert(*id, handle);
                        }
                    }
                    TaskStatus::Failed => {
                        for (d_id, d_task) in self.tasks.iter() {
                            if d_task.inputs.contains(id) {
                                self.statuses.insert(*d_id, TaskStatus::Failed);
                            }
                        }
                    }
                    TaskStatus::Done | TaskStatus::Skipped => continue,
                }
            }
            self.update_state_file()?;
        }
        Ok(())
    }
}

impl TaskQueue {
    fn update_state_file(&self) -> PyResult<()> {
        let mut all_state: ModakState = BTreeMap::new();
        for (id, task) in &self.tasks {
            let entry = TaskState {
                status: self.statuses[id].to_string(),
                inputs: task
                    .inputs
                    .iter()
                    .map(|inp_id| self.tasks[inp_id].name.clone())
                    .collect(),
                outputs: task.outputs.clone(),
                resources: task.resources.clone(),
                isolated: task.isolated,
                log_path: task.log_path.clone(),
                start_time: self.timestamps[id].0.clone(),
                end_time: self.timestamps[id].1.clone(),
            };
            all_state.insert(task.name.clone(), entry);
        }
        let json = serde_json::to_string_pretty(&all_state)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        std::fs::write(&self.state_file_path, json)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }
    fn all_done(&self) -> bool {
        self.statuses.values().all(|status| {
            matches!(
                status,
                TaskStatus::Done | TaskStatus::Skipped | TaskStatus::Failed
            )
        })
    }
    fn can_queue(&self, task: &TaskMeta) -> bool {
        for input_id in &task.inputs {
            if matches!(
                self.statuses[input_id],
                TaskStatus::Done | TaskStatus::Skipped
            ) {
                let input_task = &self.tasks[input_id];
                for output_path_str in &input_task.outputs {
                    let path = Path::new(&output_path_str);
                    if !path.exists() {
                        return false;
                    }
                }
            } else {
                return false;
            }
        }
        true
    }
    fn can_run(&self, task: &TaskMeta) -> bool {
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

#[pymodule]
fn modak(m: Bound<PyModule>) -> PyResult<()> {
    m.add_class::<TaskQueue>()?;
    Ok(())
}
