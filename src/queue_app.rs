use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};

use chrono::{DateTime, Utc};
use pyo3::{exceptions::PyValueError, PyResult};
use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Style, Stylize},
    text::Text,
    widgets::{
        Block, BorderType, Cell, Paragraph, Row, Scrollbar, ScrollbarOrientation, ScrollbarState,
        Table, TableState,
    },
    DefaultTerminal, Frame,
};

use crate::{Database, TaskRecord, TaskStatus};

const INFO_TEXT: [&str; 2] = [
    "(Esc/q) quit | (k/↑) move up | (j/↓) move down | (h/←) previous project | (l/→) next project",
    "(Enter) toggle log | (shift+k/↑) scroll to top | (shift+j/↓) scroll to bottom | (H) hide/show skipped tasks",
];

#[derive(Default)]
enum LogState {
    #[default]
    Closed,
    Open(PathBuf),
}

struct DbResult(Vec<TaskRecord>, Vec<String>);

pub struct QueueApp {
    state: TableState,
    db_tx: Sender<String>,
    result_rx: Receiver<DbResult>,
    db_pending: bool,
    current_project: usize,
    projects: Vec<String>,
    records: Vec<TaskRecord>,
    scroll_state: ScrollbarState,
    log_state: LogState,
    log_text: String,
    log_scroll_state: ScrollbarState,
    log_scroll: usize,
    log_window_lines: usize,
    follow_log: bool,
    exit: bool,
    hide_skipped: bool,
}

impl QueueApp {
    pub fn new(state_file_path: Option<PathBuf>, project: Option<String>) -> PyResult<Self> {
        let projects_from_db = Database::list_projects(state_file_path.clone())?;

        let current_project_idx = match project {
            Some(p_name) => projects_from_db
                .iter()
                .position(|p| *p == p_name)
                .ok_or_else(|| PyValueError::new_err(format!("Project '{}' not found.", p_name)))?,
            None => {
                if projects_from_db.is_empty() {
                    return Err(PyValueError::new_err(
                        "No project name provided and no projects found in the database.",
                    ));
                }
                0
            }
        };

        let (db_tx, db_rx) = mpsc::channel::<String>();
        let (result_tx, result_rx) = mpsc::channel::<DbResult>();

        let thread_state_file_path = state_file_path.clone();

        thread::spawn(move || {
            for project_command in db_rx {
                let thread_database =
                    Database::new(&project_command, thread_state_file_path.clone()).unwrap();
                let mut records = thread_database.get_project_state().unwrap();
                records.sort_by(|a, b| (a.status, b.end_time).cmp(&(b.status, a.end_time)));
                let projects = Database::list_projects(thread_state_file_path.clone()).unwrap();
                let _ = result_tx.send(DbResult(records, projects));
            }
        });

        let mut out = Self {
            state: TableState::default().with_selected(0),
            db_tx,
            result_rx,
            db_pending: false,
            current_project: current_project_idx,
            projects: projects_from_db,
            records: Vec::default(),
            scroll_state: ScrollbarState::default(),
            log_state: LogState::default(),
            log_text: String::default(),
            log_scroll_state: ScrollbarState::default(),
            log_window_lines: 0,
            log_scroll: 0,
            follow_log: true,
            exit: false,
            hide_skipped: false,
        };
        out.trigger_db_load();
        out.poll_results();
        Ok(out)
    }
    fn trigger_db_load(&mut self) {
        if !self.db_pending {
            let _ = self.db_tx.send(self.projects[self.current_project].clone());
            self.db_pending = true;
        }
    }
    fn poll_results(&mut self) {
        while let Ok(result) = self.result_rx.try_recv() {
            let current_project_name_before_update = self.projects[self.current_project].clone();
            self.records = result.0;
            self.projects = result.1;
            self.current_project = self
                .projects
                .iter()
                .position(|p| *p == current_project_name_before_update)
                .unwrap_or(0);
            self.db_pending = false;
        }
    }
    fn next_row(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i >= self.records.len() - 1 {
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
    fn bottom_row(&mut self) {
        self.state.select(Some(self.records.len() - 1));
        self.scroll_state = self.scroll_state.position(self.records.len() - 1);
    }
    fn previous_row(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i == 0 {
                    self.records.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.state.select(Some(i));
        self.scroll_state = self.scroll_state.position(i);
    }
    fn top_row(&mut self) {
        self.state.select(Some(0));
        self.scroll_state = self.scroll_state.position(0);
    }

    fn scroll_log_down(&mut self) {
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
    fn scroll_log_bottom(&mut self) {
        let max_scroll = self
            .log_text
            .lines()
            .count()
            .saturating_sub(self.log_window_lines);
        self.log_scroll = max_scroll;
        self.log_scroll_state = self.log_scroll_state.position(self.log_scroll);
        self.follow_log = true;
    }
    fn scroll_log_up(&mut self) {
        self.log_scroll = self.log_scroll.saturating_sub(1);
        self.log_scroll_state = self.log_scroll_state.position(self.log_scroll);
        self.follow_log = false;
    }
    fn scroll_log_top(&mut self) {
        self.log_scroll = 0;
        self.log_scroll_state = self.log_scroll_state.position(self.log_scroll);
        self.follow_log = false;
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> std::io::Result<()> {
        let tick_rate = Duration::from_millis(50);
        let mut last_tick = Instant::now();
        while !self.exit {
            self.poll_results();
            if let LogState::Open(path) = &self.log_state {
                self.log_text =
                    std::fs::read_to_string(path).unwrap_or("Error reading log".to_string());
            }
            terminal.draw(|frame| self.draw(frame))?;
            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));
            self.handle_events(timeout)?;
            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }
            self.trigger_db_load();
        }
        Ok(())
    }

    fn draw(&mut self, frame: &mut Frame) {
        self.scroll_state = self.scroll_state.content_length(self.records.len());
        self.log_scroll_state = self
            .log_scroll_state
            .content_length(self.log_text.lines().count());
        match &self.log_state {
            LogState::Closed => {
                let vertical = &Layout::vertical([
                    Constraint::Length(1),
                    Constraint::Fill(1),
                    Constraint::Length(4),
                ]);
                let rects = vertical.split(frame.area());
                self.render_header(frame, rects[0]);
                self.render_table(frame, rects[1]);
                self.render_scrollbar(frame, rects[1]);
                self.render_footer(frame, rects[2]);
            }
            LogState::Open(_) => {
                let vertical = &Layout::vertical([
                    Constraint::Length(1),
                    Constraint::Fill(1),
                    Constraint::Fill(1),
                    Constraint::Length(4),
                ]);
                let rects = vertical.split(frame.area());
                self.log_window_lines = rects[1].height as usize;
                self.render_header(frame, rects[0]);
                self.render_table(frame, rects[1]);
                self.render_log(frame, rects[2]);
                self.render_footer(frame, rects[3]);
            }
        }
    }
    fn handle_events(&mut self, timeout: Duration) -> std::io::Result<()> {
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    let shift_pressed = key.modifiers.contains(KeyModifiers::SHIFT);
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => self.exit = true,
                        KeyCode::Char('J') | KeyCode::Down if shift_pressed => {
                            match &self.log_state {
                                LogState::Closed => self.bottom_row(),
                                LogState::Open(_) => self.scroll_log_bottom(),
                            }
                        }
                        KeyCode::Char('K') | KeyCode::Up if shift_pressed => {
                            match &self.log_state {
                                LogState::Closed => self.top_row(),
                                LogState::Open(_) => self.scroll_log_top(),
                            }
                        }
                        KeyCode::Char('j') | KeyCode::Down => match &self.log_state {
                            LogState::Closed => self.next_row(),
                            LogState::Open(_) => self.scroll_log_down(),
                        },
                        KeyCode::Char('k') | KeyCode::Up => match &self.log_state {
                            LogState::Closed => self.previous_row(),
                            LogState::Open(_) => self.scroll_log_up(),
                        },
                        KeyCode::Char('h') | KeyCode::Left => {
                            if let LogState::Closed = self.log_state {
                                self.current_project = if self.current_project == 0 {
                                    self.projects.len() - 1
                                } else {
                                    self.current_project - 1
                                };
                            }
                        }
                        KeyCode::Char('l') | KeyCode::Right => {
                            if let LogState::Closed = self.log_state {
                                self.current_project =
                                    (self.current_project + 1) % self.projects.len();
                            }
                        }
                        KeyCode::Char('H') => {
                            self.hide_skipped = !self.hide_skipped;
                        }
                        KeyCode::Enter => match &self.log_state {
                            LogState::Closed => {
                                let log_path = self.records
                                    [self.state.selected().unwrap_or_default()]
                                .log_path
                                .clone();
                                self.log_state = LogState::Open(log_path);
                            }
                            LogState::Open(log_path) => {
                                let new_log_path = self.records
                                    [self.state.selected().unwrap_or_default()]
                                .log_path
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
        }
        Ok(())
    }
    fn render_table(&mut self, frame: &mut Frame, area: Rect) {
        let header = [
            Cell::from("Task Name"),
            Cell::from(Text::from("Status").centered()),
            Cell::from(Text::from("Start Time").centered()),
            Cell::from(Text::from("Time Elapsed").centered()),
            Cell::from(Text::from("End Time").centered()),
        ]
        .into_iter()
        .collect::<Row>()
        .height(1);
        let now = Utc::now();
        let rows: Vec<Row> = self
            .records
            .iter()
            .filter(|item| item.status != TaskStatus::Skipped || !self.hide_skipped)
            .enumerate()
            .map(|(i, item)| {
                let elapsed_time_string = if item.start_time.to_utc() > DateTime::<Utc>::default() {
                    let elapsed = (if item.end_time.to_utc() < now
                        && item.end_time.to_utc() > DateTime::<Utc>::default()
                    {
                        item.end_time.to_utc()
                    } else {
                        now
                    }) - item.start_time.to_utc();
                    let e_tot_seconds = elapsed.num_seconds();
                    let e_hours = e_tot_seconds / 3600;
                    let e_minutes = (e_tot_seconds % 3600) / 60;
                    let e_seconds = e_tot_seconds % 60;
                    format!("{:02}:{:02}:{:02}", e_hours, e_minutes, e_seconds)
                } else {
                    "--:--:--".to_string()
                };
                Row::new([
                    Cell::from(Text::from(item.name.clone())),
                    Cell::from(
                        Text::from(item.status.to_string())
                            .centered()
                            .style(Style::new().fg(item.status.color())),
                    ),
                    Cell::from(
                        Text::from(item.start_time.format("%H:%M:%S").to_string()).centered(),
                    ),
                    Cell::from(Text::from(elapsed_time_string).centered()),
                    Cell::from(Text::from(item.end_time.format("%H:%M:%S").to_string()).centered()),
                ])
                .style(Style::new().bg(if i % 2 == 0 {
                    catppuccin::PALETTE.mocha.colors.surface1.into()
                } else {
                    catppuccin::PALETTE.mocha.colors.surface2.into()
                }))
                .height(1)
            })
            .collect();
        let bar = " █ ";
        let t = Table::new(
            rows,
            [
                Constraint::Percentage(100),
                Constraint::Length(9),
                Constraint::Min(14),
                Constraint::Min(14),
                Constraint::Min(14),
            ],
        )
        .header(header)
        .highlight_symbol(Text::from(vec![bar.into()]))
        .bg(catppuccin::PALETTE.mocha.colors.base);
        frame.render_stateful_widget(t, area, &mut self.state);
        self.render_scrollbar(frame, area);
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
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(1), Constraint::Min(1)])
                    .split(area);
                if self.follow_log {
                    self.log_scroll = self
                        .log_text
                        .lines()
                        .count()
                        .saturating_sub(self.log_window_lines);
                    self.log_scroll_state = self.log_scroll_state.position(self.log_scroll);
                }
                let header = Paragraph::new(
                    self.records[self.state.selected().unwrap_or_default()]
                        .name
                        .clone(),
                )
                .style(
                    Style::new()
                        .fg(catppuccin::PALETTE.mocha.colors.base.into())
                        .bg(catppuccin::PALETTE.mocha.colors.mauve.into()),
                );
                let log = Paragraph::new(self.log_text.clone())
                    .style(
                        Style::new()
                            .fg(catppuccin::PALETTE.mocha.colors.text.into())
                            .bg(catppuccin::PALETTE.mocha.colors.surface0.into()),
                    )
                    .scroll((self.log_scroll as u16, 0));
                frame.render_widget(header, chunks[0]);
                frame.render_widget(log, chunks[1]);
                self.render_log_scrollbar(frame, chunks[1]);
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

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let time_str = Utc::now().format("%H:%M:%S UTC").to_string();
        let header_row = Row::new(vec![
            Cell::from(format!("Project: {}", self.projects[self.current_project])),
            Cell::from(Text::from(time_str).right_aligned()),
        ]);
        let table = Table::new(
            vec![header_row],
            [Constraint::Percentage(50), Constraint::Percentage(50)],
        )
        .column_spacing(1)
        .highlight_symbol("")
        .fg(catppuccin::PALETTE.mocha.colors.base)
        .bg(catppuccin::PALETTE.mocha.colors.blue);
        frame.render_widget(table, area);
    }
}
