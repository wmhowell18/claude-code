(* evaluation.mli - Position evaluation and move selection *)

open Types

(* ============================================================================
   POSITION EVALUATION
   ============================================================================ *)

(* Evaluate a single position *)
val evaluate_position :
  network_params ->
  board ->
  equity

(* Evaluate multiple positions in parallel (batched) *)
val evaluate_positions_batch :
  network_params ->
  board list ->
  equity list

(* Convert equity to single expected value *)
val equity_to_expected_value :
  equity ->
  float
  (* Returns: weighted sum considering gammon/backgammon probabilities *)


(* ============================================================================
   MOVE SELECTION
   ============================================================================ *)

(* Select best move using network evaluation *)
val select_move :
  network_params ->
  search_config ->
  board ->
  player ->
  dice ->
  search_result

(* Select move with different strategies *)
val select_move_greedy :  (* 0-ply: just pick move with best immediate equity *)
  network_params ->
  board ->
  player ->
  dice ->
  move

val select_move_1ply :  (* 1-ply lookahead (Jacob Hilton style) *)
  network_params ->
  board ->
  player ->
  dice ->
  move

val select_move_2ply :  (* 2-ply lookahead (like XG) *)
  network_params ->
  board ->
  player ->
  dice ->
  move


(* ============================================================================
   N-PLY LOOKAHEAD (GPU-optimized)
   ============================================================================ *)

(* Generic n-ply search *)
val nply_search :
  network_params ->
  board ->
  player ->
  dice ->
  int ->  (* ply_depth *)
  bool ->  (* average_over_dice (vs sample) *)
  search_result

(* 1-ply with dice averaging (most efficient for backgammon) *)
val ply1_search_averaged :
  network_params ->
  board ->
  player ->
  dice ->
  search_result

(* 2-ply with parallel GPU evaluation *)
val ply2_search_parallel :
  network_params ->
  board ->
  player ->
  dice ->
  search_result


(* ============================================================================
   MOVE EVALUATION HELPERS
   ============================================================================ *)

(* Evaluate all legal moves for a position *)
val evaluate_all_moves :
  network_params ->
  board ->
  player ->
  dice ->
  move_evaluation list

(* Evaluate single move *)
val evaluate_move :
  network_params ->
  board ->
  player ->
  move ->
  move_evaluation

(* Rank moves by equity *)
val rank_moves :
  move_evaluation list ->
  move_evaluation list


(* ============================================================================
   DICE AVERAGING (Jacob Hilton's key insight)
   ============================================================================ *)

(* Evaluate position averaged over all 21 possible opponent dice *)
val evaluate_over_opponent_dice :
  network_params ->
  board ->
  player ->
  float

(* For each legal move, evaluate resulting position over opponent dice *)
val evaluate_moves_with_dice_averaging :
  network_params ->
  board ->
  player ->
  dice ->
  move_evaluation list


(* ============================================================================
   VARIANCE REDUCTION
   ============================================================================ *)

(* Quasi-random dice sampling (better than random) *)
val quasi_random_dice_sample :
  int ->  (* num_samples *)
  rng_state ->
  dice list * rng_state

(* Stratified sampling of dice outcomes *)
val stratified_dice_sample :
  int ->  (* num_samples *)
  rng_state ->
  dice list * rng_state

(* Importance sampling for rare events *)
val importance_sample_dice :
  board ->
  player ->
  int ->  (* num_samples *)
  rng_state ->
  (dice * float) list * rng_state  (* (dice, weight) pairs *)


(* ============================================================================
   ROLLOUTS (Monte Carlo simulation)
   ============================================================================ *)

(* Play out game to completion from position *)
val rollout :
  network_params ->
  board ->
  player ->
  rng_state ->
  game_outcome * rng_state

(* Multiple parallel rollouts (GPU-optimized) *)
val rollouts_parallel :
  network_params ->
  board ->
  player ->
  int ->  (* num_rollouts *)
  rng_state ->
  game_outcome list * rng_state

(* Estimate position value via rollouts *)
val estimate_value_rollouts :
  network_params ->
  board ->
  player ->
  int ->  (* num_rollouts *)
  rng_state ->
  float * rng_state


(* ============================================================================
   MCTS (Monte Carlo Tree Search)
   ============================================================================ *)

(* MCTS node *)
type mcts_node = {
  board : board;
  player : player;
  parent : mcts_node option;

  (* Statistics *)
  visit_count : int;
  total_value : float;

  (* Children *)
  children : (dice * move * mcts_node) list;

  (* Network prior *)
  prior_value : float;
}

(* Run MCTS from position *)
val mcts_search :
  network_params ->
  board ->
  player ->
  int ->  (* num_simulations *)
  rng_state ->
  search_result * rng_state

(* Single MCTS simulation *)
val mcts_simulate :
  network_params ->
  mcts_node ->
  rng_state ->
  (mcts_node * float) * rng_state

(* UCB (Upper Confidence Bound) for node selection *)
val ucb_score :
  mcts_node ->
  float ->  (* exploration constant *)
  float

(* Backpropagate MCTS result *)
val backpropagate :
  mcts_node ->
  float ->  (* value *)
  mcts_node


(* ============================================================================
   STOCHASTIC MUZERO ADAPTATION
   ============================================================================ *)

(* Stochastic MuZero style evaluation with learned model *)
val muzero_search :
  network_params ->  (* Must include learned dynamics model *)
  board ->
  player ->
  int ->  (* num_simulations *)
  rng_state ->
  search_result * rng_state


(* ============================================================================
   PRUNING AND OPTIMIZATION
   ============================================================================ *)

(* Prune obviously bad moves early *)
val prune_dominated_moves :
  move_evaluation list ->
  move_evaluation list

(* Only evaluate top-k moves at each ply *)
val select_top_k_moves :
  move_evaluation list ->
  int ->  (* k *)
  move_evaluation list

(* Alpha-beta style pruning for backgammon *)
val alpha_beta_prune :
  network_params ->
  board ->
  player ->
  dice ->
  int ->  (* depth *)
  float ->  (* alpha *)
  float ->  (* beta *)
  search_result


(* ============================================================================
   OPENING BOOK AND ENDGAME DATABASE
   ============================================================================ *)

(* Check if position is in opening book *)
val opening_book_lookup :
  board ->
  move option

(* Check if position is in endgame database *)
val endgame_database_lookup :
  board ->
  (equity * move option) option

(* Generate endgame database for n checkers *)
val generate_endgame_database :
  int ->  (* max_checkers *)
  string ->  (* output_path *)
  unit


(* ============================================================================
   DOUBLING CUBE STRATEGY
   ============================================================================ *)

(* Cube action recommendation *)
type cube_action =
  | NoCube
  | Double
  | Take
  | Pass

(* Evaluate cube decision *)
val evaluate_cube_decision :
  network_params ->
  board ->
  player ->
  int ->  (* current_cube_value *)
  bool ->  (* can_double *)
  cube_action

(* Match equity table (for match play) *)
val match_equity :
  int ->  (* score_us *)
  int ->  (* score_them *)
  int ->  (* match_length *)
  float


(* ============================================================================
   BENCHMARKING
   ============================================================================ *)

(* Benchmark suite of positions *)
type benchmark_suite = {
  name : string;
  positions : (board * dice * move * string) list;  (* (board, dice, best_move, description) *)
}

type evaluation_results = {
  correct : int;
  total : int;
  accuracy : float;
  avg_equity_error : float;
  positions_per_second : float;
}

(* Run benchmark *)
val run_benchmark :
  network_params ->
  search_config ->
  benchmark_suite ->
  evaluation_results

(* Standard benchmark suites *)
val gnu_bg_benchmark : unit -> benchmark_suite
val xg_benchmark : unit -> benchmark_suite
val expert_positions_benchmark : unit -> benchmark_suite


(* ============================================================================
   ANALYSIS AND DEBUGGING
   ============================================================================ *)

(* Analyze a game and find mistakes *)
val analyze_game :
  network_params ->
  search_config ->
  game_record ->
  (int * move * move * float) list  (* (ply, played_move, best_move, equity_loss) *)

(* Error rate calculation *)
val calculate_error_rate :
  (int * move * move * float) list ->
  float

(* Show evaluation of all legal moves (for debugging) *)
val show_move_analysis :
  network_params ->
  board ->
  player ->
  dice ->
  string  (* Human-readable analysis *)
