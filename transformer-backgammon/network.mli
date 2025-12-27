(* network.mli - Transformer neural network architecture *)

open Types

(* ============================================================================
   NETWORK INITIALIZATION
   ============================================================================ *)

(* Create a new network with random weights *)
val init_network :
  transformer_config ->
  rng_state ->
  network_params * rng_state

(* Load network from checkpoint *)
val load_network : string -> (network_params, string) result

(* Save network to checkpoint *)
val save_network : network_params -> string -> (unit, string) result


(* ============================================================================
   FORWARD PASS
   ============================================================================ *)

(* Single forward pass: board -> equity *)
val forward :
  network_params ->
  encoded_board ->
  network_output

(* Batched forward pass (GPU-optimized) *)
val forward_batch :
  network_params ->
  encoded_board ->  (* Already batched *)
  network_output array

(* Forward pass with dropout (for training) *)
val forward_train :
  network_params ->
  encoded_board ->
  rng_state ->
  bool ->  (* training mode? *)
  network_output * rng_state


(* ============================================================================
   TRANSFORMER COMPONENTS
   ============================================================================ *)

(* Position embedding layer *)
val position_embedding :
  int ->  (* sequence length (26) *)
  int ->  (* embed_dim *)
  rng_state ->
  'params * rng_state

(* Multi-head self-attention *)
val multi_head_attention :
  'params ->
  float array array array ->  (* input: [batch, seq_len, embed_dim] *)
  int ->  (* num_heads *)
  (float array array array * float array array array array)
  (* Returns: (output, attention_weights) *)

(* Feed-forward network (used in transformer block) *)
val feed_forward :
  'params ->
  float array array array ->  (* input: [batch, seq_len, embed_dim] *)
  int ->  (* ff_dim *)
  float array array array  (* output: [batch, seq_len, embed_dim] *)

(* Single transformer block *)
val transformer_block :
  'params ->
  float array array array ->  (* input *)
  transformer_config ->
  bool ->  (* training mode *)
  rng_state ->
  (float array array array * float array array array array option) * rng_state
  (* Returns: ((output, attention_weights), rng) *)

(* Stack of transformer blocks *)
val transformer_encoder :
  'params ->
  float array array array ->  (* input *)
  transformer_config ->
  bool ->  (* training mode *)
  rng_state ->
  (float array array array * float array array array array array option) * rng_state
  (* Returns: ((output, all_attention_weights), rng) *)


(* ============================================================================
   OUTPUT HEADS
   ============================================================================ *)

(* Value head: sequence -> equity *)
val value_head :
  'params ->
  float array array array ->  (* [batch, seq_len, embed_dim] *)
  equity array  (* [batch] *)

(* Policy head: sequence -> move probabilities *)
val policy_head :
  'params ->
  float array array array ->  (* [batch, seq_len, embed_dim] *)
  int ->  (* num_actions *)
  float array array  (* [batch, num_actions] *)

(* Global pooling (for converting sequence to fixed representation) *)
val global_pool :
  float array array array ->  (* [batch, seq_len, embed_dim] *)
  string ->  (* pooling type: "mean", "max", "cls" *)
  float array array  (* [batch, embed_dim] *)


(* ============================================================================
   LOSS FUNCTIONS
   ============================================================================ *)

(* Equity loss (cross-entropy or MSE) *)
val equity_loss :
  equity array ->  (* predicted *)
  equity array ->  (* target *)
  float

(* Policy loss (cross-entropy) *)
val policy_loss :
  float array array ->  (* predicted *)
  float array array ->  (* target *)
  float

(* Combined loss *)
val total_loss :
  network_output array ->
  equity array ->  (* target equities *)
  float array array option ->  (* target policies *)
  float ->  (* equity loss weight *)
  float ->  (* policy loss weight *)
  float


(* ============================================================================
   TRAINING STEP
   ============================================================================ *)

(* Single training step: compute loss and gradients, update parameters *)
val train_step :
  network_params ->
  batch ->
  float ->  (* learning_rate *)
  'optimizer_state ->
  (network_params * 'optimizer_state * float)
  (* Returns: (updated_params, updated_opt_state, loss) *)

(* Evaluation step (no gradient computation) *)
val eval_step :
  network_params ->
  batch ->
  float
  (* Returns: loss *)


(* ============================================================================
   OPTIMIZER
   ============================================================================ *)

(* Initialize Adam optimizer *)
val init_adam_optimizer :
  float ->  (* learning_rate *)
  'optimizer_state

(* Apply Adam update *)
val adam_update :
  'optimizer_state ->
  network_params ->
  'gradients ->
  (network_params * 'optimizer_state)


(* ============================================================================
   NETWORK INSPECTION AND VISUALIZATION
   ============================================================================ *)

(* Count total parameters *)
val count_parameters : network_params -> int

(* Get parameter statistics *)
val parameter_stats : network_params -> (float * float * float)
(* Returns: (mean, std, max_abs_value) *)

(* Extract attention weights from forward pass *)
val get_attention_weights :
  network_output ->
  float array array array array option
  (* Returns: [num_layers, num_heads, seq_len, seq_len] *)

(* Visualize attention for a specific position *)
val visualize_attention :
  network_output ->
  int ->  (* layer index *)
  int ->  (* head index *)
  string
  (* Returns: ASCII visualization or path to saved image *)

(* Get embedding for a board position *)
val get_position_embedding :
  network_params ->
  encoded_board ->
  int ->  (* position index 0-25 *)
  float array


(* ============================================================================
   NETWORK ARCHITECTURES (presets)
   ============================================================================ *)

(* Small network (for testing) *)
val small_transformer_config : transformer_config

(* Medium network (baseline) *)
val medium_transformer_config : transformer_config

(* Large network (for serious training) *)
val large_transformer_config : transformer_config

(* Hybrid CNN-Transformer *)
val hybrid_config : transformer_config


(* ============================================================================
   UTILITIES
   ============================================================================ *)

(* Apply JIT compilation (JAX) *)
val jit_compile : ('a -> 'b) -> ('a -> 'b)

(* Convert network to inference mode (disable dropout, etc.) *)
val inference_mode : network_params -> network_params

(* Create EMA (exponential moving average) of parameters *)
val create_ema :
  network_params ->
  float ->  (* decay rate *)
  network_params

(* Update EMA parameters *)
val update_ema :
  network_params ->  (* current EMA *)
  network_params ->  (* new params *)
  float ->  (* decay rate *)
  network_params
