use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Display;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use chrono::{DateTime, FixedOffset, Utc};
use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::PyAny;
use ratatui::layout::{Constraint, Layout, Margin, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::Text;
use ratatui::widgets::{
    Block, BorderType, Cell, Paragraph, Row, Scrollbar, ScrollbarOrientation, ScrollbarState,
    Table, TableState,
};
use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    DefaultTerminal, Frame,
};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(u8)]
#[serde(rename_all = "lowercase")]
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
    status: TaskStatus,
    inputs: Vec<String>,
    outputs: Vec<PathBuf>,
    resources: HashMap<String, usize>,
    isolated: bool,
    log_path: PathBuf,
    start_time: String,
    end_time: String,
}

type TaskItem = (
    String,
    TaskStatus,
    DateTime<FixedOffset>,
    DateTime<FixedOffset>,
    PathBuf,
);

#[derive(Serialize, Deserialize)]
struct ModakState(BTreeMap<String, TaskState>);
impl ModakState {
    fn new() -> Self {
        Self(BTreeMap::new())
    }
}
impl From<ModakState> for Vec<TaskItem> {
    fn from(state: ModakState) -> Self {
        state
            .0
            .iter()
            .map(|(name, taskstate)| {
                (
                    name.to_owned(),
                    taskstate.status,
                    DateTime::parse_from_rfc3339(&taskstate.start_time).unwrap_or_default(),
                    DateTime::parse_from_rfc3339(&taskstate.end_time).unwrap_or_default(),
                    taskstate.log_path.clone(),
                )
            })
            .collect()
    }
}

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
        let mut all_state = ModakState::new();
        for (id, task) in &self.tasks {
            let entry = TaskState {
                status: self.statuses[id],
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
            all_state.0.insert(task.name.clone(), entry);
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

const INFO_TEXT: [&str; 2] = [
    "(Esc/q) quit | (k/↑) move up | (j/↓) move down",
    "(Enter) toggle log | (shift+k/↑) scroll to top | (shift+j/↓) scroll to bottom",
];

#[derive(Default)]
enum LogState {
    #[default]
    Closed,
    Open(PathBuf),
}

struct QueueApp {
    state: TableState,
    state_file_path: PathBuf,
    items: Vec<TaskItem>,
    scroll_state: ScrollbarState,
    log_state: LogState,
    log_text: String,
    log_scroll_state: ScrollbarState,
    log_scroll: usize,
    log_window_lines: usize,
    follow_log: bool,
    exit: bool,
}

impl QueueApp {
    fn new(state_file_path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            state: TableState::default().with_selected(0),
            state_file_path,
            items: Vec::default(),
            scroll_state: ScrollbarState::default(),
            log_state: LogState::default(),
            log_text: String::default(),
            log_scroll_state: ScrollbarState::default(),
            log_window_lines: 0,
            log_scroll: 0,
            follow_log: true,
            exit: false,
        })
    }
    fn read_state(state_file_path: &PathBuf) -> PyResult<Vec<TaskItem>> {
        let content = std::fs::read_to_string(state_file_path).map_err(PyIOError::new_err)?;
        let state: ModakState =
            serde_json::from_str(&content).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let mut state_vec: Vec<TaskItem> = state.into();
        state_vec.sort_by(|a, b| (a.1, a.3).cmp(&(b.1, b.3)));
        Ok(state_vec)
    }

    pub fn next_row(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i >= self.items.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.state.select(Some(i));
        self.scroll_state = self.scroll_state.position(i);
    }
    pub fn bottom_row(&mut self) {
        self.state.select(Some(self.items.len() - 1));
        self.scroll_state = self.scroll_state.position(self.items.len() - 1);
    }
    pub fn previous_row(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i == 0 {
                    self.items.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.state.select(Some(i));
        self.scroll_state = self.scroll_state.position(i);
    }
    pub fn top_row(&mut self) {
        self.state.select(Some(0));
        self.scroll_state = self.scroll_state.position(0);
    }

    pub fn scroll_log_down(&mut self) {
        self.log_scroll = self.log_scroll.saturating_add(1);
        let max_scroll = self
            .log_text
            .lines()
            .count()
            .saturating_sub(self.log_window_lines);
        if self.log_scroll > max_scroll {
            self.log_scroll = max_scroll;
            self.follow_log = true;
        }
        self.log_scroll_state = self.log_scroll_state.position(self.log_scroll);
    }
    pub fn scroll_log_bottom(&mut self) {
        let max_scroll = self
            .log_text
            .lines()
            .count()
            .saturating_sub(self.log_window_lines);
        self.log_scroll = max_scroll;
        self.log_scroll_state = self.log_scroll_state.position(self.log_scroll);
        self.follow_log = true;
    }
    pub fn scroll_log_up(&mut self) {
        self.log_scroll = self.log_scroll.saturating_sub(1);
        self.log_scroll_state = self.log_scroll_state.position(self.log_scroll);
        self.follow_log = false;
    }
    pub fn scroll_log_top(&mut self) {
        self.log_scroll = 0;
        self.log_scroll_state = self.log_scroll_state.position(self.log_scroll);
        self.follow_log = false;
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> std::io::Result<()> {
        while !self.exit {
            if let Ok(updated_items) = Self::read_state(&self.state_file_path) {
                self.items = updated_items;
            }
            if let LogState::Open(path) = &self.log_state {
                self.log_text =
                    std::fs::read_to_string(path).unwrap_or("Error reading log".to_string());
            }
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events()?;
        }
        Ok(())
    }

    fn draw(&mut self, frame: &mut Frame) {
        self.scroll_state = self.scroll_state.content_length(self.items.len());
        self.log_scroll_state = self
            .log_scroll_state
            .content_length(self.log_text.lines().count());
        match &self.log_state {
            LogState::Closed => {
                let vertical = &Layout::vertical([Constraint::Fill(1), Constraint::Length(4)]);
                let rects = vertical.split(frame.area());
                self.render_table(frame, rects[0]);
                self.render_scrollbar(frame, rects[0]);
                self.render_footer(frame, rects[1]);
            }
            LogState::Open(_) => {
                let vertical = &Layout::vertical([
                    Constraint::Fill(1),
                    Constraint::Fill(1),
                    Constraint::Length(4),
                ]);
                let rects = vertical.split(frame.area());
                self.log_window_lines = rects[1].height as usize;
                self.render_table(frame, rects[0]);
                self.render_scrollbar(frame, rects[0]);
                self.render_log(frame, rects[1]);
                self.render_log_scrollbar(frame, rects[1]);
                self.render_footer(frame, rects[2]);
            }
        }
    }
    fn handle_events(&mut self) -> std::io::Result<()> {
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                let shift_pressed = key.modifiers.contains(KeyModifiers::SHIFT);
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => self.exit = true,
                    KeyCode::Char('J') | KeyCode::Down if shift_pressed => match &self.log_state {
                        LogState::Closed => self.bottom_row(),
                        LogState::Open(_) => self.scroll_log_bottom(),
                    },
                    KeyCode::Char('K') | KeyCode::Up if shift_pressed => match &self.log_state {
                        LogState::Closed => self.top_row(),
                        LogState::Open(_) => self.scroll_log_top(),
                    },
                    KeyCode::Char('j') | KeyCode::Down => match &self.log_state {
                        LogState::Closed => self.next_row(),
                        LogState::Open(_) => self.scroll_log_down(),
                    },
                    KeyCode::Char('k') | KeyCode::Up => match &self.log_state {
                        LogState::Closed => self.previous_row(),
                        LogState::Open(_) => self.scroll_log_up(),
                    },
                    KeyCode::Enter => match &self.log_state {
                        LogState::Closed => {
                            let log_path = self.items[self.state.selected().unwrap_or_default()]
                                .4
                                .clone();
                            self.log_state = LogState::Open(log_path);
                        }
                        LogState::Open(log_path) => {
                            let new_log_path = self.items
                                [self.state.selected().unwrap_or_default()]
                            .4
                            .clone();
                            if *log_path != new_log_path {
                                self.log_state = LogState::Open(new_log_path);
                            } else {
                                self.log_state = LogState::Closed;
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        Ok(())
    }
    fn render_table(&mut self, frame: &mut Frame, area: Rect) {
        let header = ["Task Name", "Status", "Start Time", "End Time"]
            .into_iter()
            .map(Cell::from)
            .collect::<Row>()
            .height(1);
        let rows: Vec<Row> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                Row::new([
                    Cell::from(Text::from(item.0.clone())),
                    Cell::from(
                        Text::from(item.1.to_string()).style(Style::new().fg(item.1.color())),
                    ),
                    Cell::from(item.2.format("%H:%M:%S").to_string()),
                    Cell::from(item.3.format("%H:%M:%S").to_string()),
                ])
                .style(Style::new().bg(if i % 2 == 0 {
                    catppuccin::PALETTE.mocha.colors.surface0.into()
                } else {
                    catppuccin::PALETTE.mocha.colors.surface1.into()
                }))
                .height(1)
            })
            .collect();
        let bar = " █ ";
        let t = Table::new(
            rows,
            [
                Constraint::Min(20),
                Constraint::Length(9),
                Constraint::Min(12),
                Constraint::Min(12),
            ],
        )
        .header(header)
        .highlight_symbol(Text::from(vec![bar.into()]))
        .bg(catppuccin::PALETTE.mocha.colors.base);
        frame.render_stateful_widget(t, area, &mut self.state);
    }
    fn render_scrollbar(&mut self, frame: &mut Frame, area: Rect) {
        frame.render_stateful_widget(
            Scrollbar::default()
                .orientation(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None),
            area.inner(Margin {
                vertical: 1,
                horizontal: 1,
            }),
            &mut self.scroll_state,
        );
    }
    fn render_log(&mut self, frame: &mut Frame, area: Rect) {
        match &self.log_state {
            LogState::Closed => {}
            LogState::Open(_) => {
                if self.follow_log {
                    self.log_scroll = self
                        .log_text
                        .lines()
                        .count()
                        .saturating_sub(self.log_window_lines);
                    self.log_scroll_state = self.log_scroll_state.position(self.log_scroll);
                }
                let paragraph =
                    Paragraph::new(self.log_text.clone()).scroll((self.log_scroll as u16, 0));
                frame.render_widget(paragraph, area);
            }
        }
    }
    fn render_log_scrollbar(&mut self, frame: &mut Frame, area: Rect) {
        frame.render_stateful_widget(
            Scrollbar::default()
                .orientation(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None),
            area.inner(Margin {
                vertical: 1,
                horizontal: 1,
            }),
            &mut self.log_scroll_state,
        );
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let info_footer = Paragraph::new(Text::from_iter(INFO_TEXT))
            .centered()
            .block(Block::bordered().border_type(BorderType::Double));
        frame.render_widget(info_footer, area);
    }
}

#[pyfunction]
fn run_queue_wrapper(state_file_path: PathBuf) -> PyResult<()> {
    let mut terminal = ratatui::init();
    let result = QueueApp::new(state_file_path)?.run(&mut terminal);
    ratatui::restore();
    result.map_err(|e| PyIOError::new_err(e.to_string()))
}

#[pymodule]
fn modak(m: Bound<PyModule>) -> PyResult<()> {
    m.add_class::<TaskQueue>()?;
    m.add_function(wrap_pyfunction!(run_queue_wrapper, &m)?)?;
    Ok(())
}
