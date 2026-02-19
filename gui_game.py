"""
=============================================================================
  Tic-Tac-Toe | Minimax AI Agent  –  GUI Edition
=============================================================================
  Course  : Artificial Intelligence
  Program : Master of Science in Artificial Intelligence
  Topic   : Adversarial Search – Minimax Algorithm
  Instructor: Dr. Natnael Argaw Wondimu
-----------------------------------------------------------------------------
  This single-file program contains:
    1. Game Representation  (TicTacToe class)
    2. Minimax Agent        (MinimaxAgent class – with Alpha-Beta pruning)
    3. GUI Interface        (TicTacToeGUI class – tkinter)
=============================================================================
  Run:  python gui_game.py
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Section 1 – Imports
# ─────────────────────────────────────────────────────────────────────────────
import math
import time
import tkinter as tk
from tkinter import messagebox, font as tkfont
from copy import deepcopy
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Section 2 – Game Representation
# ─────────────────────────────────────────────────────────────────────────────
class TicTacToe:
    """
    Formal game representation for Tic-Tac-Toe.

    Attributes
    ----------
    board          : 3×3 grid  – each cell is ' ', 'X', or 'O'
    current_player : whose turn it is ('X' or 'O')

    Game properties satisfied
    -------------------------
    ✔ Deterministic        – no randomness
    ✔ Turn-based           – players alternate
    ✔ Zero-sum             – one player's gain is the other's loss
    ✔ Perfect information  – full board is always visible
    """

    def __init__(self):
        """Initialize an empty 3×3 board; X always moves first."""
        self.board: List[List[str]] = [[' '] * 3 for _ in range(3)]
        self.current_player: str = 'X'

    # ── State helpers ────────────────────────────────────────────────────────

    def get_board(self) -> List[List[str]]:
        """Return a deep copy of the current board (immutable view)."""
        return deepcopy(self.board)

    def clone(self) -> 'TicTacToe':
        """Create a full deep copy of this game state for search."""
        new = TicTacToe()
        new.board = deepcopy(self.board)
        new.current_player = self.current_player
        return new

    # ── Legal actions ────────────────────────────────────────────────────────

    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """
        Return all legal moves as (row, col) pairs.
        A move is legal iff the cell is empty (' ').
        """
        return [(r, c) for r in range(3) for c in range(3)
                if self.board[r][c] == ' ']

    def is_valid_move(self, row: int, col: int) -> bool:
        """True iff (row, col) is in-bounds and the cell is empty."""
        return 0 <= row <= 2 and 0 <= col <= 2 and self.board[row][col] == ' '

    # ── Move execution ───────────────────────────────────────────────────────

    def make_move(self, row: int, col: int) -> bool:
        """
        Place the current player's symbol at (row, col) and switch turns.
        Returns False if the move was illegal.
        """
        if not self.is_valid_move(row, col):
            return False
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    # ── Terminal-state detection ─────────────────────────────────────────────

    def is_terminal(self) -> bool:
        """True when the game has ended (win or draw)."""
        return self.check_winner() is not None or not self.get_legal_actions()

    def check_winner(self) -> Optional[str]:
        """
        Inspect every winning line on the board.

        Returns
        -------
        'X'    – X has three in a row
        'O'    – O has three in a row
        'Draw' – board is full, no winner
        None   – game is still in progress
        """
        b = self.board

        # Rows
        for row in b:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]

        # Columns
        for c in range(3):
            if b[0][c] == b[1][c] == b[2][c] != ' ':
                return b[0][c]

        # Diagonals
        if b[0][0] == b[1][1] == b[2][2] != ' ':
            return b[0][0]
        if b[0][2] == b[1][1] == b[2][0] != ' ':
            return b[0][2]

        # Draw
        if not self.get_legal_actions():
            return 'Draw'

        return None

    def get_winning_line(self) -> Optional[List[Tuple[int, int]]]:
        """Return the list of (row,col) cells that form the winning line, or None."""
        b = self.board
        lines = [
            [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],  # rows
            [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],  # cols
            [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)],                         # diags
        ]
        for line in lines:
            symbols = [b[r][c] for r, c in line]
            if symbols[0] == symbols[1] == symbols[2] != ' ':
                return line
        return None

    # ── Utility function ─────────────────────────────────────────────────────

    def utility(self, player: str) -> int:
        """
        Numerical outcome from the perspective of *player*.

        Win  → +10   (player has three in a row)
        Loss → -10   (opponent has three in a row)
        Draw →   0   (board full, no winner)
        """
        winner = self.check_winner()
        if winner == player:
            return 10
        if winner == 'Draw':
            return 0
        if winner is not None:
            return -10
        return 0  # non-terminal (should not reach here in normal usage)

    def get_result_message(self) -> str:
        """Human-readable summary of the game outcome."""
        w = self.check_winner()
        if w == 'Draw':
            return "It's a Draw!"
        if w:
            return f"Player {w} Wins!"
        return "Game in Progress"


# ─────────────────────────────────────────────────────────────────────────────
#  Section 3 – Minimax Agent
# ─────────────────────────────────────────────────────────────────────────────
class MinimaxAgent:
    """
    Intelligent agent implementing the Minimax decision rule.

    The agent explores the complete game tree and always selects the move
    that maximises its minimum guaranteed utility, assuming the opponent
    also plays optimally.

    Optional enhancement: Alpha-Beta pruning eliminates branches that
    cannot affect the final decision, reducing nodes expanded significantly.
    """

    def __init__(self, player: str, use_alpha_beta: bool = True):
        """
        Parameters
        ----------
        player        : symbol this agent controls ('X' or 'O')
        use_alpha_beta: whether to enable α-β pruning (default: True)
        """
        self.player = player
        self.opponent = 'O' if player == 'X' else 'X'
        self.use_alpha_beta = use_alpha_beta
        self.nodes_expanded: int = 0

    # ── Public interface ─────────────────────────────────────────────────────

    def get_best_move(self, game: TicTacToe) -> Tuple[int, int]:
        """
        Compute the optimal move for the current board state.

        Returns
        -------
        (row, col) of the best action found by Minimax search.
        """
        self.nodes_expanded = 0
        if self.use_alpha_beta:
            _, move = self._ab_max(game, -math.inf, math.inf)
        else:
            _, move = self._max(game)
        return move

    def get_nodes_expanded(self) -> int:
        """Number of game-tree nodes visited in the last search call."""
        return self.nodes_expanded

    # ── Standard Minimax ─────────────────────────────────────────────────────

    def _max(self, game: TicTacToe) -> Tuple[int, Optional[Tuple[int, int]]]:
        """MAX node: choose the action with the highest utility."""
        self.nodes_expanded += 1
        if game.is_terminal():
            return game.utility(self.player), None

        best_val, best_move = -math.inf, None
        for action in game.get_legal_actions():
            child = game.clone()
            child.make_move(*action)
            # Opponent is always MIN from this agent's view
            val, _ = self._min(child)
            if val > best_val:
                best_val, best_move = val, action
        return best_val, best_move

    def _min(self, game: TicTacToe) -> Tuple[int, Optional[Tuple[int, int]]]:
        """MIN node: choose the action with the lowest utility (worst for agent)."""
        self.nodes_expanded += 1
        if game.is_terminal():
            return game.utility(self.player), None

        best_val, best_move = math.inf, None
        for action in game.get_legal_actions():
            child = game.clone()
            child.make_move(*action)
            val, _ = self._max(child)
            if val < best_val:
                best_val, best_move = val, action
        return best_val, best_move

    # ── Alpha-Beta Pruning ───────────────────────────────────────────────────

    def _ab_max(self, game: TicTacToe,
                alpha: float, beta: float) -> Tuple[int, Optional[Tuple[int, int]]]:
        """MAX node with α-β pruning."""
        self.nodes_expanded += 1
        if game.is_terminal():
            return game.utility(self.player), None

        best_val, best_move = -math.inf, None
        for action in game.get_legal_actions():
            child = game.clone()
            child.make_move(*action)
            val, _ = self._ab_min(child, alpha, beta)
            if val > best_val:
                best_val, best_move = val, action
            if best_val >= beta:          # β cut-off
                return best_val, best_move
            alpha = max(alpha, best_val)
        return best_val, best_move

    def _ab_min(self, game: TicTacToe,
                alpha: float, beta: float) -> Tuple[int, Optional[Tuple[int, int]]]:
        """MIN node with α-β pruning."""
        self.nodes_expanded += 1
        if game.is_terminal():
            return game.utility(self.player), None

        best_val, best_move = math.inf, None
        for action in game.get_legal_actions():
            child = game.clone()
            child.make_move(*action)
            val, _ = self._ab_max(child, alpha, beta)
            if val < best_val:
                best_val, best_move = val, action
            if best_val <= alpha:         # α cut-off
                return best_val, best_move
            beta = min(beta, best_val)
        return best_val, best_move


# ─────────────────────────────────────────────────────────────────────────────
#  Section 4 – GUI Interface
# ─────────────────────────────────────────────────────────────────────────────

# ── Colour palette (dark premium theme) ──────────────────────────────────────
COLORS = {
    "bg":          "#0D1117",   # Deep navy-black background
    "panel":       "#161B22",   # Slightly lighter panel
    "cell_empty":  "#1C2333",   # Cell unfilled
    "cell_hover":  "#21273A",   # Cell hover highlight
    "x_color":     "#58A6FF",   # Vibrant blue for X
    "o_color":     "#F78166",   # Coral-orange for O
    "win_flash":   "#388BFD",   # Flash color for winning line
    "accent":      "#3FB950",   # Green accent
    "text_main":   "#E6EDF3",   # Primary text
    "text_dim":    "#8B949E",   # Dimmed /secondary text
    "border":      "#30363D",   # Subtle border
    "btn_primary": "#238636",   # Primary button (green)
    "btn_hover":   "#2EA043",
    "btn_danger":  "#DA3633",
    "btn_danger_h":"#F85149",
    "btn_blue":    "#1F6FEB",
    "btn_blue_h":  "#388BFD",
    "title_grad1": "#58A6FF",
    "title_grad2": "#F78166",
}


class TicTacToeGUI:
    """
    Full-featured graphical interface for Tic-Tac-Toe vs. Minimax AI.

    Features
    --------
    ✔ Dark premium UI
    ✔ Animated hover effects on cells
    ✔ Winning-line flash animation
    ✔ Symbol selection (X / O)
    ✔ First-player selection (Human / AI)
    ✔ Alpha-Beta pruning toggle
    ✔ Live performance metrics (nodes expanded, time taken)
    ✔ Move history log
    ✔ Play-again without restarting the program
    """

    WIN  = "\U0001F3C6"   # 🏆
    DRAW = "\U0001F91D"   # 🤝
    LOSE = "\U0001F916"   # 🤖

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Tic-Tac-Toe  ·  Minimax AI Agent")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(False, False)

        # ── Fonts ────────────────────────────────────────────────────────────
        self.font_title  = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self.font_sub    = tkfont.Font(family="Segoe UI", size=10)
        self.font_cell   = tkfont.Font(family="Segoe UI", size=48, weight="bold")
        self.font_label  = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.font_small  = tkfont.Font(family="Segoe UI", size=9)
        self.font_btn    = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.font_log    = tkfont.Font(family="Consolas", size=9)
        self.font_status = tkfont.Font(family="Segoe UI", size=12, weight="bold")

        # ── State variables ──────────────────────────────────────────────────
        self.human_symbol   = tk.StringVar(value='X')
        self.first_player   = tk.StringVar(value='Human')
        self.use_alpha_beta = tk.BooleanVar(value=True)

        self.game: Optional[TicTacToe] = None
        self.agent: Optional[MinimaxAgent] = None
        self.human_player: str = 'X'
        self.ai_player: str = 'O'
        self.game_active: bool = False
        self.move_count: int = 0

        self.total_nodes: int = 0
        self.total_ai_time: float = 0.0
        self.ai_move_count: int = 0

        self._flash_job = None   # after() handle for win animation
        self._ai_job    = None   # after() handle for AI delay

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_ui()
        self._center_window(900, 660)

    # ── Window helpers ───────────────────────────────────────────────────────

    def _center_window(self, w: int, h: int):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        """Assemble all panels of the interface."""
        outer = tk.Frame(self.root, bg=COLORS["bg"])
        outer.pack(fill="both", expand=True, padx=18, pady=14)

        # ── Header ────────────────────────────────────────────────────────────
        self._build_header(outer)

        # ── Content row (left panel + board + right panel) ────────────────────
        content = tk.Frame(outer, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, pady=(8, 0))

        self._build_left_panel(content)
        self._build_board(content)
        self._build_right_panel(content)

    def _build_header(self, parent):
        hdr = tk.Frame(parent, bg=COLORS["bg"])
        hdr.pack(fill="x", pady=(0, 4))

        tk.Label(hdr, text="Tic-Tac-Toe", font=self.font_title,
                 fg=COLORS["x_color"], bg=COLORS["bg"]).pack(side="left")
        tk.Label(hdr, text="  ×  ", font=self.font_title,
                 fg=COLORS["text_dim"], bg=COLORS["bg"]).pack(side="left")
        tk.Label(hdr, text="Minimax AI Agent", font=self.font_title,
                 fg=COLORS["o_color"], bg=COLORS["bg"]).pack(side="left")

        badge = tk.Label(hdr,
                         text=" AI  ",
                         font=self.font_small,
                         fg=COLORS["bg"], bg=COLORS["accent"],
                         padx=6, pady=2)
        badge.pack(side="left", padx=10, pady=6)

        tk.Label(hdr,
                 text="MSc Artificial Intelligence  ·  Adversarial Search",
                 font=self.font_small, fg=COLORS["text_dim"], bg=COLORS["bg"]
                 ).pack(side="right", padx=4)

        sep = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep.pack(fill="x", pady=(0, 8))

    def _build_left_panel(self, parent):
        """Configuration controls on the left."""
        left = tk.Frame(parent, bg=COLORS["panel"],
                        bd=0, relief="flat", width=200)
        left.pack(side="left", fill="y", padx=(0, 12), pady=0)
        left.pack_propagate(False)

        def section(title):
            tk.Frame(left, bg=COLORS["border"], height=1).pack(
                fill="x", padx=10, pady=(14, 4))
            tk.Label(left, text=title, font=self.font_small,
                     fg=COLORS["text_dim"], bg=COLORS["panel"]).pack(
                anchor="w", padx=14)

        tk.Label(left, text="⚙  Settings", font=self.font_label,
                 fg=COLORS["text_main"], bg=COLORS["panel"]
                 ).pack(anchor="w", padx=14, pady=(14, 0))

        # ── Symbol ─────────────────────────────────────────────────────
        section("Your Symbol")
        sym_row = tk.Frame(left, bg=COLORS["panel"])
        sym_row.pack(fill="x", padx=14, pady=4)
        for sym, col in [('X', COLORS["x_color"]), ('O', COLORS["o_color"])]:
            rb = tk.Radiobutton(sym_row, text=f" {sym}", variable=self.human_symbol,
                                value=sym, font=self.font_label,
                                fg=col, bg=COLORS["panel"],
                                selectcolor=COLORS["cell_empty"],
                                activebackground=COLORS["panel"],
                                activeforeground=col,
                                cursor="hand2")
            rb.pack(side="left", padx=6)

        # ── First player ───────────────────────────────────────────────
        section("First Move")
        fp_row = tk.Frame(left, bg=COLORS["panel"])
        fp_row.pack(fill="x", padx=14, pady=4)
        for fp in ['Human', 'AI']:
            rb = tk.Radiobutton(fp_row, text=f" {fp}", variable=self.first_player,
                                value=fp, font=self.font_label,
                                fg=COLORS["text_main"], bg=COLORS["panel"],
                                selectcolor=COLORS["cell_empty"],
                                activebackground=COLORS["panel"],
                                activeforeground=COLORS["accent"],
                                cursor="hand2")
            rb.pack(side="left", padx=4)

        # ── Alpha-Beta ─────────────────────────────────────────────────
        section("Algorithm")
        self.ab_check = tk.Checkbutton(
            left, text=" Alpha-Beta Pruning",
            variable=self.use_alpha_beta,
            font=self.font_small,
            fg=COLORS["accent"], bg=COLORS["panel"],
            selectcolor=COLORS["cell_empty"],
            activebackground=COLORS["panel"],
            activeforeground=COLORS["accent"],
            cursor="hand2")
        self.ab_check.pack(anchor="w", padx=14, pady=4)

        # ── Buttons ────────────────────────────────────────────────────
        tk.Frame(left, bg=COLORS["border"], height=1).pack(
            fill="x", padx=10, pady=12)

        self.start_btn = self._make_btn(left, "▶  New Game",
                                        COLORS["btn_primary"], COLORS["btn_hover"],
                                        self._start_game)
        self.start_btn.pack(fill="x", padx=14, pady=3)

        self.reset_btn = self._make_btn(left, "↺  Reset Board",
                                        COLORS["btn_blue"], COLORS["btn_blue_h"],
                                        self._reset_board)
        self.reset_btn.pack(fill="x", padx=14, pady=3)
        self.reset_btn.config(state="disabled")

        quit_btn = self._make_btn(left, "✕  Quit",
                                  COLORS["btn_danger"], COLORS["btn_danger_h"],
                                  self.root.quit)
        quit_btn.pack(fill="x", padx=14, pady=3)

        # ── Algorithm info ─────────────────────────────────────────────
        tk.Frame(left, bg=COLORS["border"], height=1).pack(
            fill="x", padx=10, pady=(16, 4))
        info = (
            "Minimax explores the\n"
            "full game tree to find\n"
            "the optimal move.\n\n"
            "α-β pruning skips\n"
            "branches that cannot\n"
            "affect the outcome,\n"
            "reducing search cost."
        )
        tk.Label(left, text=info, font=self.font_small,
                 fg=COLORS["text_dim"], bg=COLORS["panel"],
                 justify="left", wraplength=170
                 ).pack(anchor="w", padx=14, pady=4)

    def _build_board(self, parent):
        """Centre panel: status bar + 3×3 cell grid."""
        centre = tk.Frame(parent, bg=COLORS["bg"])
        centre.pack(side="left", fill="both", expand=True)

        # Status label
        self.status_var = tk.StringVar(value="Configure settings and press  ▶ New Game")
        self.status_lbl = tk.Label(centre, textvariable=self.status_var,
                                   font=self.font_status,
                                   fg=COLORS["accent"], bg=COLORS["bg"],
                                   pady=6)
        self.status_lbl.pack(pady=(0, 6))

        # Board frame
        board_frame = tk.Frame(centre, bg=COLORS["border"], bd=2, relief="flat")
        board_frame.pack()

        self.cells: List[List[tk.Label]] = []
        for r in range(3):
            row_widgets = []
            for c in range(3):
                cell = tk.Label(board_frame,
                                text='',
                                font=self.font_cell,
                                width=3, height=1,
                                bg=COLORS["cell_empty"],
                                fg=COLORS["text_main"],
                                relief="flat",
                                cursor="hand2")
                cell.grid(row=r, column=c,
                          padx=3, pady=3,
                          ipadx=10, ipady=10)
                cell.bind("<Button-1>", lambda e, row=r, col=c: self._on_cell_click(row, col))
                cell.bind("<Enter>",    lambda e, row=r, col=c: self._on_hover(row, col, True))
                cell.bind("<Leave>",    lambda e, row=r, col=c: self._on_hover(row, col, False))
                row_widgets.append(cell)
            self.cells.append(row_widgets)

        # Turn indicator bar
        turn_frame = tk.Frame(centre, bg=COLORS["bg"])
        turn_frame.pack(pady=(10, 0))
        tk.Label(turn_frame, text="Turn:", font=self.font_small,
                 fg=COLORS["text_dim"], bg=COLORS["bg"]).pack(side="left")
        self.x_turn_lbl = tk.Label(turn_frame, text="  X  ",
                                    font=self.font_label,
                                    fg=COLORS["bg"], bg=COLORS["x_color"],
                                    padx=8, pady=2)
        self.x_turn_lbl.pack(side="left", padx=4)
        self.o_turn_lbl = tk.Label(turn_frame, text="  O  ",
                                    font=self.font_label,
                                    fg=COLORS["bg"], bg=COLORS["text_dim"],
                                    padx=8, pady=2)
        self.o_turn_lbl.pack(side="left", padx=4)

    def _build_right_panel(self, parent):
        """Right panel: performance metrics + move history log."""
        right = tk.Frame(parent, bg=COLORS["panel"], width=210)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        tk.Label(right, text="📊  Metrics", font=self.font_label,
                 fg=COLORS["text_main"], bg=COLORS["panel"]
                 ).pack(anchor="w", padx=14, pady=(14, 4))

        # Metric labels
        self.metric_vars: dict = {}
        metrics = [
            ("Moves Made",      "moves",       "0"),
            ("Nodes Expanded",  "nodes",       "0"),
            ("AI Time (ms)",    "ai_time",     "0.00"),
            ("Avg Nodes/Move",  "avg_nodes",   "—"),
            ("Algorithm",       "algorithm",   "Minimax + α-β"),
        ]
        for label, key, default in metrics:
            row = tk.Frame(right, bg=COLORS["panel"])
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=label + ":", font=self.font_small,
                     fg=COLORS["text_dim"], bg=COLORS["panel"],
                     anchor="w", width=16).pack(side="left")
            var = tk.StringVar(value=default)
            self.metric_vars[key] = var
            tk.Label(row, textvariable=var, font=self.font_small,
                     fg=COLORS["accent"], bg=COLORS["panel"],
                     anchor="e").pack(side="right")

        tk.Frame(right, bg=COLORS["border"], height=1).pack(
            fill="x", padx=10, pady=(12, 4))

        tk.Label(right, text="📋  Move History", font=self.font_label,
                 fg=COLORS["text_main"], bg=COLORS["panel"]
                 ).pack(anchor="w", padx=14, pady=(0, 4))

        log_frame = tk.Frame(right, bg=COLORS["panel"])
        log_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        scrollbar = tk.Scrollbar(log_frame, bg=COLORS["panel"])
        scrollbar.pack(side="right", fill="y")

        self.log_box = tk.Text(log_frame,
                               font=self.font_log,
                               bg=COLORS["cell_empty"],
                               fg=COLORS["text_main"],
                               insertbackground=COLORS["text_main"],
                               relief="flat", bd=0,
                               state="disabled",
                               width=20,
                               yscrollcommand=scrollbar.set)
        self.log_box.pack(fill="both", expand=True)
        scrollbar.config(command=self.log_box.yview)

        self.log_box.tag_config("human", foreground=COLORS["x_color"])
        self.log_box.tag_config("ai",    foreground=COLORS["o_color"])
        self.log_box.tag_config("info",  foreground=COLORS["text_dim"])
        self.log_box.tag_config("win",   foreground=COLORS["accent"])

    # ── Widget helpers ───────────────────────────────────────────────────────

    def _make_btn(self, parent, text, bg, hover_bg, command):
        btn = tk.Label(parent, text=text, font=self.font_btn,
                       fg=COLORS["text_main"], bg=bg,
                       padx=10, pady=7,
                       cursor="hand2", relief="flat")
        btn.bind("<Button-1>", lambda e: command() if str(btn.cget("state")) != "disabled" else None)
        btn.bind("<Enter>",    lambda e: btn.config(bg=hover_bg))
        btn.bind("<Leave>",    lambda e: btn.config(bg=bg))
        # Store colours for enable/disable
        btn._normal_bg = bg
        btn._hover_bg  = hover_bg
        return btn

    # ── Game lifecycle ───────────────────────────────────────────────────────

    def _start_game(self):
        """Initialise a fresh game with the selected configuration."""
        # Cancel any pending after() calls
        if self._flash_job:
            self.root.after_cancel(self._flash_job)
            self._flash_job = None
        if self._ai_job:
            self.root.after_cancel(self._ai_job)
            self._ai_job = None

        self.human_player = self.human_symbol.get()
        self.ai_player    = 'O' if self.human_player == 'X' else 'X'

        self.game        = TicTacToe()
        self.agent       = MinimaxAgent(self.ai_player, self.use_alpha_beta.get())
        self.game_active = True
        self.move_count  = 0

        self.total_nodes    = 0
        self.total_ai_time  = 0.0
        self.ai_move_count  = 0

        # Respect first-player setting
        first = self.first_player.get()
        if first == 'AI' and self.game.current_player == self.human_player:
            # X goes first by default; if human is O and AI should go first we're fine
            # If human is X and AI should go first we have an issue — just swap concept:
            # We'll re-assign current_player for the starting board
            self.game.current_player = self.ai_player
        elif first == 'Human' and self.game.current_player != self.human_player:
            self.game.current_player = self.human_player

        # Reset board visuals
        for r in range(3):
            for c in range(3):
                self.cells[r][c].config(text='', bg=COLORS["cell_empty"],
                                        fg=COLORS["text_main"])

        # Reset metrics
        self._update_metrics()
        self._update_algorithm_label()

        # Reset log
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")
        self._log("── New Game Started ──", "info")

        # Buttons
        self.reset_btn.config(state="normal", bg=COLORS["btn_blue"])
        self.start_btn.config(bg=COLORS["btn_primary"])

        # Block settings during game
        self.ab_check.config(state="disabled")

        self._update_status_and_turn()

        # If AI goes first, trigger AI move
        if self.game.current_player == self.ai_player:
            self._schedule_ai_move()

    def _reset_board(self):
        """Quick-reset: same config, fresh board."""
        self._start_game()

    def _end_game(self):
        """Handle terminal state — announce result and unlock settings."""
        self.game_active = False
        self.ab_check.config(state="normal")

        result = self.game.get_result_message()
        winner = self.game.check_winner()

        # Highlight winning cells
        win_line = self.game.get_winning_line()
        if win_line:
            self._flash_winning_line(win_line, 0)

        # Status message
        if winner == self.human_player:
            icon = self.WIN
            self.status_var.set(f"{icon}  You Win!  {icon}")
            self.status_lbl.config(fg=COLORS["accent"])
            self._log(f"🏆 You Win! Congratulations!", "win")
        elif winner == self.ai_player:
            icon = self.LOSE
            self.status_var.set(f"{icon}  AI Wins!  {icon}")
            self.status_lbl.config(fg=COLORS["o_color"])
            self._log(f"🤖 AI Wins! Minimax played optimally.", "ai")
        else:
            icon = self.DRAW
            self.status_var.set(f"{icon}  Draw!  {icon}")
            self.status_lbl.config(fg=COLORS["text_dim"])
            self._log(f"🤝 It's a Draw!", "info")

        self._log(f"Total moves: {self.move_count}", "info")
        self._log(f"Total nodes expanded: {self.total_nodes}", "info")

    # ── Cell interaction ─────────────────────────────────────────────────────

    def _on_cell_click(self, row: int, col: int):
        if not self.game_active:
            return
        if self.game.current_player != self.human_player:
            return
        if not self.game.is_valid_move(row, col):
            self._shake_cell(row, col)
            return

        self._place_move(row, col)

        if self.game.is_terminal():
            self._end_game()
        else:
            self._update_status_and_turn()
            self._schedule_ai_move()

    def _on_hover(self, row: int, col: int, entering: bool):
        if not self.game_active:
            return
        if self.game.current_player != self.human_player:
            return
        if self.game.board[row][col] != ' ':
            return
        self.cells[row][col].config(
            bg=COLORS["cell_hover"] if entering else COLORS["cell_empty"])

    def _place_move(self, row: int, col: int):
        """Execute a move and update cell visual."""
        player = self.game.current_player
        self.game.make_move(row, col)
        self.move_count += 1

        color = COLORS["x_color"] if player == 'X' else COLORS["o_color"]
        self.cells[row][col].config(text=player, fg=color, bg=COLORS["cell_empty"])

        is_human = (player == self.human_player)
        tag  = "human" if is_human else "ai"
        who  = "You" if is_human else "AI"
        self._log(f"#{self.move_count:>2}  {who} ({player}) → ({row},{col})", tag)
        self._update_metrics()

    # ── AI scheduling ─────────────────────────────────────────────────────────

    def _schedule_ai_move(self, delay_ms: int = 400):
        """Schedule the AI move slightly after the human move for UX."""
        self._ai_job = self.root.after(delay_ms, self._do_ai_move)

    def _do_ai_move(self):
        if not self.game_active or self.game.is_terminal():
            return
        if self.game.current_player != self.ai_player:
            return

        self.status_var.set("🤖  AI is thinking…")
        self.status_lbl.config(fg=COLORS["o_color"])
        self.root.update_idletasks()

        t0  = time.perf_counter()
        row, col = self.agent.get_best_move(self.game)
        dt  = (time.perf_counter() - t0) * 1000  # ms

        nodes = self.agent.get_nodes_expanded()
        self.total_nodes   += nodes
        self.total_ai_time += dt
        self.ai_move_count += 1

        self._place_move(row, col)
        self._update_metrics(last_nodes=nodes, last_time=dt)

        if self.game.is_terminal():
            self._end_game()
        else:
            self._update_status_and_turn()

    # ── Visual effects ────────────────────────────────────────────────────────

    def _flash_winning_line(self, line, count):
        """Alternate golden/blue colours on winning cells."""
        colors_on  = [COLORS["accent"], "#2EA043", COLORS["accent"]]
        colors_off = [COLORS["cell_empty"]] * 3

        for idx, (r, c) in enumerate(line):
            bg = colors_on[idx % len(colors_on)] if count % 2 == 0 else colors_off[idx % len(colors_off)]
            self.cells[r][c].config(bg=bg)

        if count < 8:
            self._flash_job = self.root.after(
                280, self._flash_winning_line, line, count + 1)

    def _shake_cell(self, row: int, col: int):
        """Brief red flash on an invalid click."""
        original_bg = self.cells[row][col].cget("bg")
        self.cells[row][col].config(bg=COLORS["btn_danger"])
        self.root.after(200, lambda: self.cells[row][col].config(bg=original_bg))

    # ── Status / metrics helpers ──────────────────────────────────────────────

    def _update_status_and_turn(self):
        cp = self.game.current_player
        is_human = (cp == self.human_player)

        if is_human:
            self.status_var.set(f"Your turn  ({self.human_player})  — click a cell")
            self.status_lbl.config(fg=COLORS["x_color"] if self.human_player == 'X'
                                   else COLORS["o_color"])
        else:
            self.status_var.set(f"AI's turn  ({self.ai_player})…")
            self.status_lbl.config(fg=COLORS["o_color"] if self.ai_player == 'O'
                                   else COLORS["x_color"])

        # Turn indicator bar
        x_bg = COLORS["x_color"] if cp == 'X' else COLORS["text_dim"]
        o_bg = COLORS["o_color"] if cp == 'O' else COLORS["text_dim"]
        self.x_turn_lbl.config(bg=x_bg)
        self.o_turn_lbl.config(bg=o_bg)

    def _update_metrics(self, last_nodes: int = None, last_time: float = None):
        self.metric_vars["moves"].set(str(self.move_count))
        self.metric_vars["nodes"].set(f"{self.total_nodes:,}")
        if last_time is not None:
            self.metric_vars["ai_time"].set(f"{last_time:.2f}")
        if self.ai_move_count > 0:
            avg = self.total_nodes / self.ai_move_count
            self.metric_vars["avg_nodes"].set(f"{avg:.1f}")

    def _update_algorithm_label(self):
        label = "Minimax + α-β" if self.use_alpha_beta.get() else "Minimax (no pruning)"
        self.metric_vars["algorithm"].set(label)

    def _log(self, message: str, tag: str = "info"):
        self.log_box.config(state="normal")
        self.log_box.insert("end", message + "\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────
#  Section 5 – Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
