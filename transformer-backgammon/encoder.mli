(* encoder.mli - Board encoding for neural network input *)

open Types

(* ============================================================================
   BOARD ENCODING
   ============================================================================ *)

(* Encode a single board state into neural network input format *)
val encode_board : encoding_config -> board -> encoded_board

(* Encode a batch of boards *)
val encode_boards : encoding_config -> board list -> encoded_board

(* Decode board from network representation (for debugging) *)
val decode_board : encoding_config -> encoded_board -> board option


(* ============================================================================
   POSITION FEATURE EXTRACTION
   ============================================================================ *)

(* Extract features for a single position (one of the 26 positions) *)
val extract_position_features :
  encoding_config ->
  board ->
  point ->  (* Which position (0=bar, 1-24=points, 25=off) *)
  float array

(* Feature dimension for a given config *)
val feature_dimension : encoding_config -> int


(* ============================================================================
   ENCODING VARIANTS
   ============================================================================ *)

(* Raw encoding: just checker counts *)
val encode_raw :
  board ->
  point ->
  float array
(* Returns: [our_checkers, opp_checkers] - length 2 *)

(* One-hot encoding: checker counts as one-hot vectors *)
val encode_one_hot :
  board ->
  point ->
  float array
(* Returns: [our_checkers_onehot (16 dims), opp_checkers_onehot (16 dims)] - length 32 *)

(* Geometric encoding: add positional information *)
val encode_geometric :
  board ->
  point ->
  float array
(* Returns: raw features + [distance_to_home, is_home_board, is_opp_home] - length 5 *)

(* Strategic encoding: add hand-crafted strategic features *)
val encode_strategic :
  board ->
  point ->
  float array
(* Returns: geometric + [is_blot, is_anchor, is_prime, hit_prob, ...] - length ~10-15 *)

(* Full encoding: everything (configurable) *)
val encode_full :
  encoding_config ->
  board ->
  point ->
  float array


(* ============================================================================
   DICE ENCODING (optional)
   ============================================================================ *)

(* Encode dice roll as features *)
val encode_dice : dice -> float array
(* Returns: [die1_onehot (6 dims), die2_onehot (6 dims)] - length 12 *)

(* Encode dice as embedding ID *)
val dice_to_embedding_id : dice -> int
(* Returns: 0-20 (one of 21 unique dice outcomes) *)


(* ============================================================================
   EQUITY ENCODING (for training targets)
   ============================================================================ *)

(* Convert game outcome to equity target *)
val outcome_to_equity : game_outcome -> player -> equity

(* Encode equity as array for training *)
val equity_to_array : equity -> float array
(* Returns: [win_normal, win_gammon, win_backgammon, lose_gammon, lose_backgammon] *)

(* Decode equity from array *)
val array_to_equity : float array -> equity


(* ============================================================================
   NORMALIZATION AND PREPROCESSING
   ============================================================================ *)

(* Normalize features (mean=0, std=1) *)
val normalize_features : float array array -> float array array

(* Standardize board representation *)
(* Always represent from perspective of player to move *)
val canonicalize_board : board -> board

(* Data augmentation: create equivalent positions *)
val augment_position : board -> board list
(* E.g., flip colors, mirror board (if applicable) *)


(* ============================================================================
   ENCODING CONFIGURATIONS (presets)
   ============================================================================ *)

(* Minimal encoding (like TD-Gammon raw) *)
val minimal_encoding_config : encoding_config

(* Standard encoding (like Jacob Hilton's 326 inputs) *)
val standard_encoding_config : encoding_config

(* Rich encoding (everything) *)
val rich_encoding_config : encoding_config

(* Pure raw encoding (most general, let transformer learn everything) *)
val raw_encoding_config : encoding_config


(* ============================================================================
   BATCH UTILITIES
   ============================================================================ *)

(* Convert list of encoded boards to batched tensor representation *)
val stack_encoded_boards : encoded_board list -> encoded_board

(* Shuffle a batch (for training) *)
val shuffle_batch : batch -> rng_state -> batch * rng_state

(* Split batch into minibatches *)
val split_batch : batch -> int -> batch list
