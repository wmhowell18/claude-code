(* main.mli - Main entry points for the application *)

open Types

(* ============================================================================
   MAIN COMMANDS
   ============================================================================ *)

(* Train a new model from scratch *)
val train_from_scratch :
  config ->
  unit

(* Continue training from checkpoint *)
val train_from_checkpoint :
  config ->
  string ->  (* checkpoint_path *)
  unit

(* Evaluate a trained model *)
val evaluate_model :
  string ->  (* model_path *)
  player_agent ->  (* opponent *)
  int ->  (* num_games *)
  unit

(* Play interactively against the model *)
val play_interactive :
  string ->  (* model_path *)
  search_config ->
  unit

(* Analyze a game *)
val analyze_game_file :
  string ->  (* model_path *)
  string ->  (* game_file *)
  search_config ->
  unit

(* Run benchmarks *)
val run_benchmarks :
  string ->  (* model_path *)
  search_config ->
  string ->  (* benchmark_name *)
  unit


(* ============================================================================
   EXPERIMENTATION
   ============================================================================ *)

(* Compare different architectures *)
val compare_architectures :
  transformer_config list ->
  training_config ->
  unit

(* Hyperparameter search *)
val hyperparameter_search :
  config ->
  string ->  (* search_type: "grid", "random", "bayesian" *)
  int ->  (* num_trials *)
  unit

(* Ablation study *)
val ablation_study :
  config ->
  string list ->  (* features to ablate *)
  unit


(* ============================================================================
   UTILITIES
   ============================================================================ *)

(* Self-play and generate training data *)
val generate_training_data :
  string ->  (* model_path *)
  int ->  (* num_games *)
  string ->  (* output_path *)
  unit

(* Convert between formats *)
val convert_checkpoint :
  string ->  (* input_path *)
  string ->  (* output_path *)
  string ->  (* output_format: "jax", "pytorch", "onnx" *)
  unit

(* Export model for deployment *)
val export_model :
  string ->  (* model_path *)
  string ->  (* output_path *)
  string ->  (* format: "jax", "onnx", "tflite" *)
  unit


(* ============================================================================
   DEMO AND VISUALIZATION
   ============================================================================ *)

(* Demonstrate model playing *)
val demo :
  string ->  (* model_path *)
  int ->  (* num_games *)
  bool ->  (* verbose *)
  unit

(* Visualize attention patterns *)
val visualize_attention :
  string ->  (* model_path *)
  board ->
  string ->  (* output_path *)
  unit

(* Visualize learning progress *)
val visualize_training :
  string ->  (* log_dir *)
  string ->  (* output_path *)
  unit


(* ============================================================================
   INTEGRATION WITH EXTERNAL PROGRAMS
   ============================================================================ *)

(* Interface with GNU Backgammon *)
val gnubg_interface :
  string ->  (* model_path *)
  unit

(* Export to XG format *)
val export_to_xg :
  string ->  (* model_path *)
  string ->  (* output_path *)
  unit


(* ============================================================================
   COMMAND-LINE INTERFACE
   ============================================================================ *)

(* Parse command-line arguments and dispatch *)
val main : string array -> unit

(* Print help message *)
val print_help : unit -> unit

(* Print version *)
val print_version : unit -> unit


(* ============================================================================
   SERVERS AND APIS
   ============================================================================ *)

(* Start inference server *)
val start_server :
  string ->  (* model_path *)
  int ->  (* port *)
  unit

(* REST API endpoints *)
val api_evaluate_position :
  string ->  (* JSON request *)
  string  (* JSON response *)

val api_select_move :
  string ->  (* JSON request *)
  string  (* JSON response *)

val api_analyze_game :
  string ->  (* JSON request *)
  string  (* JSON response *)
