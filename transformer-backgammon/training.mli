(* training.mli - Training loop and experience replay *)

open Types

(* ============================================================================
   EXPERIENCE REPLAY
   ============================================================================ *)

(* Create empty replay buffer *)
val create_replay_buffer : int -> replay_buffer

(* Add a training example to the buffer *)
val add_to_buffer : replay_buffer -> training_example -> replay_buffer

(* Add multiple examples *)
val add_batch_to_buffer : replay_buffer -> training_example list -> replay_buffer

(* Sample a random minibatch from buffer *)
val sample_batch :
  replay_buffer ->
  int ->  (* batch_size *)
  rng_state ->
  batch option * rng_state

(* Convert game record to training examples *)
val game_to_training_examples :
  game_record ->
  network_params option ->  (* Optional: use network for intermediate evals *)
  training_example list

(* Add game to replay buffer *)
val add_game_to_buffer :
  replay_buffer ->
  game_record ->
  replay_buffer


(* ============================================================================
   TRAINING LOOP
   ============================================================================ *)

(* Main training loop *)
val train :
  config ->
  network_params option ->  (* Starting params, or None for random init *)
  unit
  (* Side effects: trains network, saves checkpoints, logs metrics *)

(* Single training iteration *)
val training_iteration :
  network_params ->
  'optimizer_state ->
  replay_buffer ->
  training_config ->
  rng_state ->
  (network_params * 'optimizer_state * replay_buffer * training_metrics) * rng_state

(* Training step on one minibatch *)
val training_minibatch :
  network_params ->
  'optimizer_state ->
  batch ->
  training_config ->
  (network_params * 'optimizer_state * float)
  (* Returns: (updated_params, updated_opt, loss) *)


(* ============================================================================
   SELF-PLAY DATA GENERATION
   ============================================================================ *)

(* Generate self-play games for training *)
val generate_selfplay_games :
  network_params ->
  search_config ->
  int ->  (* num_games *)
  rng_state ->
  game_record list * rng_state

(* Parallel self-play (GPU-optimized) *)
val generate_selfplay_parallel :
  network_params ->
  search_config ->
  int ->  (* num_games *)
  int ->  (* num_parallel_workers *)
  rng_state ->
  game_record list * rng_state


(* ============================================================================
   EVALUATION
   ============================================================================ *)

(* Evaluate network against opponent *)
val evaluate :
  network_params ->
  player_agent ->  (* opponent *)
  int ->  (* num_games *)
  rng_state ->
  (float * training_metrics) * rng_state
  (* Returns: (win_rate, metrics) *)

(* Evaluate on a benchmark suite *)
val evaluate_benchmark :
  network_params ->
  benchmark_suite ->  (* Forward reference - suite of known positions *)
  evaluation_results

(* Compare two networks head-to-head *)
val compare_networks :
  network_params ->
  network_params ->
  int ->  (* num_games *)
  rng_state ->
  float * rng_state
  (* Returns: win_rate of first network vs second *)


(* ============================================================================
   CURRICULUM LEARNING
   ============================================================================ *)

(* Progressive difficulty training *)
type curriculum_stage = {
  stage_name : string;
  opponent : player_agent;
  num_games : int;
  search_depth : int;
  success_threshold : float;  (* Win rate to advance *)
}

(* Train through curriculum *)
val train_curriculum :
  config ->
  curriculum_stage list ->
  network_params option ->
  network_params


(* ============================================================================
   TRAINING VARIANTS
   ============================================================================ *)

(* TD(λ) learning (like original TD-Gammon) *)
val train_td_lambda :
  config ->
  float ->  (* λ parameter *)
  network_params option ->
  network_params

(* Jacob Hilton's approach: TD(0) with dice averaging *)
val train_td_zero_averaged :
  config ->
  network_params option ->
  network_params

(* AlphaZero-style: MCTS + self-play *)
val train_alphazero_style :
  config ->
  int ->  (* MCTS simulations per move *)
  network_params option ->
  network_params


(* ============================================================================
   CHECKPOINTING
   ============================================================================ *)

(* Save training checkpoint *)
val save_checkpoint :
  network_params ->
  'optimizer_state ->
  replay_buffer ->
  training_metrics ->
  string ->  (* checkpoint_path *)
  (unit, string) result

(* Load training checkpoint *)
val load_checkpoint :
  string ->  (* checkpoint_path *)
  (network_params * 'optimizer_state * replay_buffer * training_metrics, string) result

(* Save best model *)
val save_best_model :
  network_params ->
  float ->  (* current_win_rate *)
  string ->  (* model_dir *)
  unit


(* ============================================================================
   LOGGING AND MONITORING
   ============================================================================ *)

(* Log training metrics *)
val log_metrics :
  training_metrics ->
  unit

(* Log to Weights & Biases *)
val log_to_wandb :
  training_metrics ->
  network_params option ->  (* Optional: log model *)
  unit

(* Print training progress *)
val print_progress :
  training_metrics ->
  unit

(* Create training summary *)
val create_summary :
  training_metrics list ->
  string  (* Returns formatted summary *)


(* ============================================================================
   HYPERPARAMETER OPTIMIZATION
   ============================================================================ *)

(* Hyperparameter search space *)
type hyperparam_space = {
  learning_rates : float list;
  batch_sizes : int list;
  network_sizes : transformer_config list;
  (* ... etc *)
}

(* Run hyperparameter search *)
val hyperparameter_search :
  hyperparam_space ->
  int ->  (* num_trials *)
  (config * float)  (* Returns: (best_config, best_score) *)


(* ============================================================================
   UTILITIES
   ============================================================================ *)

(* Calculate learning rate with warmup and decay *)
val learning_rate_schedule :
  int ->  (* current_step *)
  int ->  (* warmup_steps *)
  int ->  (* total_steps *)
  float ->  (* base_lr *)
  float

(* Early stopping checker *)
val should_stop_early :
  training_metrics list ->
  int ->  (* patience *)
  bool

(* Compute TD(λ) returns for a game *)
val compute_td_returns :
  game_record ->
  float ->  (* λ *)
  float ->  (* γ (discount factor) *)
  float list  (* Returns for each position *)
