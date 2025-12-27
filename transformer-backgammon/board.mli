(* board.mli - Board representation and game rules *)

open Types

(* ============================================================================
   BOARD CONSTRUCTION AND MANIPULATION
   ============================================================================ *)

(* Create initial board state (standard backgammon starting position) *)
val initial_board : unit -> board

(* Create empty board *)
val empty_board : unit -> board

(* Create board from position description (for testing/debugging) *)
val board_from_string : string -> (board, string) result

(* Clone a board *)
val copy_board : board -> board

(* Flip board perspective (white <-> black) *)
val flip_board : board -> board


(* ============================================================================
   BOARD QUERIES
   ============================================================================ *)

(* Get number of checkers at a point for a player *)
val get_checkers : board -> player -> point -> checker_count

(* Set number of checkers at a point for a player *)
val set_checkers : board -> player -> point -> checker_count -> board

(* Total pip count for a player (sum of point_number * checkers) *)
val pip_count : board -> player -> int

(* Is the game over? *)
val is_game_over : board -> bool

(* Who won? Returns None if game not over *)
val winner : board -> game_outcome option

(* Can player bear off? (all checkers in home board) *)
val can_bear_off : board -> player -> bool

(* Number of checkers on the bar *)
val checkers_on_bar : board -> player -> int

(* Number of checkers borne off *)
val checkers_borne_off : board -> player -> int


(* ============================================================================
   MOVE GENERATION
   ============================================================================ *)

(* Generate all legal moves for a board + dice + player *)
val generate_legal_moves : board -> player -> dice -> legal_moves

(* Is a specific move legal? *)
val is_legal_move : board -> player -> dice -> move -> bool

(* Get all legal single-step moves for one die value *)
val generate_single_die_moves : board -> player -> int -> move_step list

(* Helper: can a checker move from point A to point B? *)
val can_move_checker : board -> player -> point -> point -> bool


(* ============================================================================
   MOVE APPLICATION
   ============================================================================ *)

(* Apply a move to a board, returning new board *)
val apply_move : board -> player -> move -> (board, string) result

(* Apply a single move step *)
val apply_move_step : board -> player -> move_step -> (board, string) result

(* Undo a move (for search/rollback) *)
val undo_move : board -> player -> move -> (board, string) result


(* ============================================================================
   DICE UTILITIES
   ============================================================================ *)

(* All possible dice rolls (21 unique outcomes) *)
val all_dice_rolls : unit -> dice list

(* Is this a doubles roll? *)
val is_doubles : dice -> bool

(* Get the dice values to use (doubles -> 4 values, normal -> 2) *)
val dice_values : dice -> int list

(* Sample a random dice roll *)
val roll_dice : rng_state -> dice * rng_state


(* ============================================================================
   GAME SIMULATION
   ============================================================================ *)

(* Play one ply: given board, player, and dice, choose and apply move *)
(* Returns new board and the move chosen *)
val play_ply :
  board ->
  player ->
  dice ->
  (move -> float) ->  (* Move evaluation function *)
  (board * move, string) result

(* Simulate a complete game between two players *)
val simulate_game :
  player_agent ->  (* White player *)
  player_agent ->  (* Black player *)
  rng_state ->
  game_record * rng_state

(* Simulate N games in parallel (GPU-friendly) *)
val simulate_games_parallel :
  player_agent ->
  player_agent ->
  int ->  (* Number of games *)
  rng_state ->
  game_record list * rng_state


(* ============================================================================
   BOARD VALIDATION AND DEBUGGING
   ============================================================================ *)

(* Check if a board state is valid *)
val is_valid_board : board -> (bool, string) result

(* Pretty-print board to string *)
val board_to_string : board -> string

(* Print board to terminal with formatting *)
val print_board : board -> unit

(* Export board to GNU Backgammon position ID *)
val to_gnubg_position_id : board -> string

(* Import board from GNU Backgammon position ID *)
val from_gnubg_position_id : string -> (board, string) result


(* ============================================================================
   POSITION FEATURES (for hand-crafted features if needed)
   ============================================================================ *)

(* Strategic feature extraction (useful for debugging or hybrid approaches) *)

(* Does player have an anchor at point? *)
val has_anchor : board -> player -> point -> bool

(* Strength of blockade (number of consecutive made points) *)
val blockade_strength : board -> player -> int

(* Number of blots (vulnerable single checkers) *)
val count_blots : board -> player -> int

(* Probability that a blot at point gets hit *)
val hit_probability : board -> point -> float

(* Is this a racing position? (no contact between players) *)
val is_racing_position : board -> bool

(* Count of checkers in home board *)
val checkers_in_home : board -> player -> int

(* Count of back checkers (opponent's home board) *)
val back_checkers : board -> player -> int
