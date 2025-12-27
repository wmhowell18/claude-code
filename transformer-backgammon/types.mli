(* types.mli - Core type definitions for transformer backgammon *)

(* ============================================================================
   BOARD REPRESENTATION
   ============================================================================ *)

(* Position on the board: 1-24 are points, 0 is bar, 25 is off *)
type point = int  (* 0..25 *)

(* Player representation *)
type player = White | Black

(* Number of checkers on a point (0-15) *)
type checker_count = int  (* 0..15 *)

(* Board state: for each point, how many checkers of each color *)
type board = {
  white_checkers : checker_count array;  (* length 26: [bar, 1..24, off] *)
  black_checkers : checker_count array;  (* length 26: [bar, 1..24, off] *)
  player_to_move : player;
}

(* Dice roll *)
type dice = int * int  (* (die1, die2) where 1 <= die1, die2 <= 6 *)

(* A single checker movement *)
type move_step = {
  from_point : point;
  to_point : point;
  die_used : int;
  hits_opponent : bool;  (* Does this move hit an opponent blot? *)
}

(* A complete move (may use 1-4 dice for doubles) *)
type move = move_step list

(* Legal moves for a given board + dice *)
type legal_moves = move list


(* ============================================================================
   NEURAL NETWORK REPRESENTATIONS
   ============================================================================ *)

(* Tensor shape type for documentation *)
type shape = int list  (* e.g., [batch_size, seq_len, embed_dim] *)

(* Board encoded as features for neural network input *)
type encoded_board = {
  (* Position sequence: [batch, 26, feature_dim] *)
  (* 26 = bar + 24 points + off *)
  position_features : float array array array;

  (* Optional: dice encoding if conditioning on dice *)
  dice_features : float array option;

  (* Shape metadata for validation *)
  batch_size : int;
  sequence_length : int;  (* Always 26 for our case *)
  feature_dim : int;
}

(* Network output: equity estimation *)
type equity = {
  win_normal : float;        (* P(win without gammon) *)
  win_gammon : float;        (* P(win with gammon) *)
  win_backgammon : float;    (* P(win with backgammon) *)
  lose_gammon : float;       (* P(lose with gammon) *)
  lose_backgammon : float;   (* P(lose with backgammon) *)
}
(* Note: P(lose normal) = 1 - sum of above *)

(* Combined network output *)
type network_output = {
  equity : equity;

  (* Optional policy head for move probabilities *)
  move_policy : float array option;  (* [num_legal_moves] *)

  (* Optional: attention weights for interpretability *)
  attention_weights : float array array array option;  (* [num_layers, num_heads, 26, 26] *)
}

(* Network parameters (weights) *)
type network_params = {
  (* Opaque type - actual structure depends on JAX/Flax implementation *)
  params : 'a;  (* Will be PyTree in JAX *)

  (* Metadata *)
  num_parameters : int;
  architecture_config : config;  (* Forward reference to config type *)
}


(* ============================================================================
   TRAINING DATA STRUCTURES
   ============================================================================ *)

(* A single training example *)
type training_example = {
  board_state : board;
  player : player;

  (* Target values from game outcome *)
  target_equity : equity;

  (* Optional: policy target from MCTS or n-ply search *)
  target_policy : float array option;

  (* Game context *)
  game_id : int;
  ply_number : int;
}

(* Experience replay buffer *)
type replay_buffer = {
  examples : training_example array;  (* Circular buffer *)
  capacity : int;
  current_size : int;
  write_index : int;
}

(* Training batch *)
type batch = {
  boards : encoded_board;  (* [batch_size, 26, feature_dim] *)
  target_equities : equity array;  (* [batch_size] *)
  target_policies : float array array option;  (* Optional: [batch_size, num_moves] *)
}

(* Training metrics *)
type training_metrics = {
  epoch : int;
  total_games : int;
  total_positions : int;

  (* Loss values *)
  equity_loss : float;
  policy_loss : float option;
  total_loss : float;

  (* Performance metrics *)
  win_rate_vs_random : float option;
  win_rate_vs_baseline : float option;

  (* Training speed *)
  positions_per_second : float;
  games_per_minute : float;
}


(* ============================================================================
   EVALUATION AND SEARCH
   ============================================================================ *)

(* Search configuration for move selection *)
type search_config = {
  ply_depth : int;  (* 0 = just network eval, 1 = 1-ply lookahead, etc. *)

  (* Whether to average over all dice (like Jacob Hilton) or sample *)
  average_dice : bool;
  num_dice_samples : int option;  (* If not averaging, how many samples? *)

  (* Pruning: only consider top-k moves at each ply *)
  prune_to_top_k : int option;
}

(* Evaluation result for a move *)
type move_evaluation = {
  move : move;
  equity : equity;
  expected_equity : float;  (* Single number: weighted combination *)

  (* Debugging info *)
  evaluations_count : int;  (* How many positions evaluated for this move *)
}

(* Result of searching for best move *)
type search_result = {
  best_move : move;
  best_equity : float;

  (* All evaluated moves, sorted by equity *)
  all_moves : move_evaluation list;

  (* Search statistics *)
  positions_evaluated : int;
  time_ms : float;
}


(* ============================================================================
   GAME SIMULATION
   ============================================================================ *)

(* Game outcome *)
type game_outcome =
  | WhiteWins of int  (* Points: 1=normal, 2=gammon, 3=backgammon *)
  | BlackWins of int

(* Complete game record *)
type game_record = {
  game_id : int;

  (* Position history *)
  positions : (board * dice * move) list;

  (* Final outcome *)
  outcome : game_outcome;

  (* Metadata *)
  num_plies : int;
  white_player : string;  (* Player name/type *)
  black_player : string;
  timestamp : float;
}

(* Player type for self-play *)
type player_agent =
  | NetworkAgent of network_params * search_config
  | RandomAgent
  | GreedyAgent of network_params  (* 0-ply, just pick best immediate eval *)
  | ExternalAgent of string  (* GNU BG, XG, etc. via API *)


(* ============================================================================
   CONFIGURATION
   ============================================================================ *)

(* Transformer architecture configuration *)
type transformer_config = {
  (* Architecture *)
  num_layers : int;
  embed_dim : int;
  num_heads : int;
  ff_dim : int;  (* Feed-forward hidden dimension *)

  (* Regularization *)
  dropout_rate : float;
  layer_norm_epsilon : float;

  (* Input/output *)
  input_feature_dim : int;
  use_learned_positional_encoding : bool;

  (* Heads *)
  use_policy_head : bool;
  return_attention_weights : bool;  (* For visualization *)
}

(* Board encoding configuration *)
type encoding_config = {
  (* What features to include per position *)
  use_one_hot_counts : bool;  (* vs continuous counts *)
  include_geometric_features : bool;  (* distance to home, etc. *)
  include_strategic_features : bool;  (* is_blot, is_anchor, etc. *)
  include_dice_encoding : bool;

  (* Resulting feature dimension *)
  feature_dim : int;
}

(* Training configuration *)
type training_config = {
  (* Optimizer *)
  learning_rate : float;
  optimizer : string;  (* "adam", "sgd", etc. *)

  (* Training loop *)
  batch_size : int;
  num_epochs : int;
  minibatches_per_epoch : int;

  (* Experience replay *)
  replay_buffer_size : int;
  min_replay_size : int;  (* Minimum before training starts *)

  (* Self-play *)
  games_per_iteration : int;
  selfplay_search_config : search_config;

  (* Evaluation *)
  eval_every_n_games : int;
  eval_num_games : int;
  eval_opponent : player_agent;

  (* Checkpointing *)
  checkpoint_every_n_games : int;
  checkpoint_dir : string;
}

(* Complete configuration *)
type config = {
  transformer : transformer_config;
  encoding : encoding_config;
  training : training_config;

  (* System *)
  seed : int;
  device : string;  (* "cpu", "gpu", "tpu" *)
  use_jit : bool;  (* JAX JIT compilation *)

  (* Logging *)
  log_level : string;  (* "debug", "info", "warning", "error" *)
  log_dir : string;
  use_wandb : bool;  (* Weights & Biases integration *)
  wandb_project : string;
}


(* ============================================================================
   UTILITY TYPES
   ============================================================================ *)

(* Result type for error handling *)
type ('a, 'e) result =
  | Ok of 'a
  | Error of 'e

(* Random number generator state *)
type rng_state = 'a  (* JAX PRNGKey or similar *)
