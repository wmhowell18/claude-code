(* cube.mli - Doubling cube for match play *)

open Types

(* ============================================================================
   CUBE STATE
   ============================================================================ *)

(* Cube value: 1, 2, 4, 8, 16, 32, 64 *)
type cube_value = int

(* Cube ownership *)
type cube_owner =
  | Centered       (* Neither player owns the cube *)
  | OwnedBy of player

(* Complete cube state *)
type cube_state = {
  value : cube_value;
  owner : cube_owner;
}

(* Initial cube state (centered, value 1) *)
val initial_cube : unit -> cube_state

(* Maximum cube value *)
val max_cube_value : int  (* 64 *)


(* ============================================================================
   CUBE DECISIONS
   ============================================================================ *)

(* Possible cube actions *)
type cube_action =
  | NoDouble          (* Don't double *)
  | Double            (* Offer a double *)
  | Take              (* Accept the double *)
  | Pass              (* Decline the double (forfeit game) *)
  | Beaver            (* Accept and immediately redouble - optional rule *)

(* Result of a cube action *)
type cube_decision_result =
  | ContinueGame of cube_state      (* Game continues with new cube state *)
  | GameOver of player * int        (* Player wins, points *)


(* ============================================================================
   CUBE RULES
   ============================================================================ *)

(* Can a player double? *)
val can_double : cube_state -> player -> bool

(* Apply a cube action *)
val apply_cube_action :
  cube_state ->
  player ->
  cube_action ->
  (cube_decision_result, string) result

(* Get legal cube actions for a player *)
val legal_cube_actions : cube_state -> player -> cube_action list


(* ============================================================================
   MATCH PLAY
   ============================================================================ *)

(* Match state (for playing to N points) *)
type match_state = {
  target_points : int;        (* Match length *)
  white_score : int;
  black_score : int;
  crawford : bool;            (* Crawford rule: no doubling in next game after someone reaches match point *)
  post_crawford : bool;       (* Are we in post-Crawford games? *)
}

(* Create new match *)
val new_match : int -> match_state

(* Is match over? *)
val is_match_over : match_state -> bool

(* Get match winner *)
val match_winner : match_state -> player option

(* Update match score after a game *)
val update_match_score :
  match_state ->
  player ->     (* winner *)
  int ->        (* points won *)
  match_state

(* Crawford rule: should cube be disabled? *)
val crawford_game : match_state -> bool


(* ============================================================================
   CUBE EQUITY
   ============================================================================ *)

(* Cube equity: expected value considering cube state *)
type cube_equity = {
  raw_equity : equity;           (* Equity without cube *)
  cubeful_equity : float;        (* Equity accounting for cube *)
  double_equity : float;         (* Equity if we double *)
  take_equity : float;           (* Equity if opponent takes *)
  pass_equity : float;           (* Equity if opponent passes *)
}

(* Calculate cubeful equity *)
val calculate_cube_equity :
  equity ->
  cube_state ->
  player ->
  match_state option ->  (* None for money game *)
  cube_equity

(* Should we double? (compares no-double vs double equity) *)
val should_double : cube_equity -> bool

(* Should we take? (compares take vs pass equity) *)
val should_take : cube_equity -> bool


(* ============================================================================
   CUBE HANDLING IN NETWORK
   ============================================================================ *)

(* Encode cube state as features for neural network *)
val encode_cube_state : cube_state -> float array
(* Returns: [cube_value_normalized, is_centered, we_own_cube, opp_owns_cube] *)

(* Decode cube decision from network output *)
val decode_cube_decision : float array -> cube_action

(* Cube decision head output shape *)
val cube_decision_output_dim : int  (* 4: no_double, double, take, pass *)


(* ============================================================================
   CUBE TRAINING
   ============================================================================ *)

(* Generate cube decision training example from game *)
val cube_training_example :
  game_record ->
  cube_state ->
  int ->        (* ply number *)
  (board * cube_state * cube_action * float) option
  (* Returns: (position, cube_state, correct_action, equity) if cube decision occurred *)

(* Cube decision quality metrics *)
type cube_decision_quality = {
  equity_error : float;           (* How much equity we lost/gained *)
  was_blunder : bool;             (* Lost > 0.1 equity *)
  was_correct : bool;             (* Within 0.02 equity of optimal *)
}

val evaluate_cube_decision :
  cube_action ->
  cube_equity ->
  cube_decision_quality


(* ============================================================================
   CUBE AGENT
   ============================================================================ *)

(* Agent that can make cube decisions *)
type cube_agent = {
  base_agent : agent;  (* For move selection *)
  cube_decision : board -> cube_state -> match_state option -> cube_action;
}

(* Create cube agent from regular agent with heuristic cube decisions *)
val heuristic_cube_agent : agent -> cube_agent

(* Create cube agent from network that outputs cube decisions *)
val network_cube_agent :
  network_params ->
  encoding_config ->
  string ->
  cube_agent


(* ============================================================================
   REFERENCE CUBE DECISIONS
   ============================================================================ *)

(* Reference cube actions based on pip count and equity *)
type reference_cube_decision = {
  gammon_threat : float;         (* Probability of gammon *)
  volatility : float;            (* Position volatility *)
  recommended_action : cube_action;
}

(* Simple reference cube decision (for training baseline) *)
val reference_cube_decision :
  board ->
  equity ->
  cube_state ->
  reference_cube_decision
