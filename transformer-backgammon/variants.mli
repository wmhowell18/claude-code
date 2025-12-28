(* variants.mli - Board position variants for training diversity *)

open Types

(* ============================================================================
   STARTING POSITION VARIANTS
   ============================================================================ *)

(* Standard backgammon starting position *)
val standard_start : unit -> board

(* Nackgammon starting position
   Variation: 2 checkers on 23-point and 24-point instead of both on 24 *)
val nackgammon_start : unit -> board

(* Hypergammon starting position
   Minimal variant: 3 checkers each (1 on 24, 23, 22 for white; 1, 2, 3 for black) *)
val hypergammon_start : unit -> board

(* Long gammon starting position (31-point board instead of 24) *)
val long_gammon_start : unit -> board


(* ============================================================================
   PERTURBED STARTING POSITIONS
   ============================================================================ *)

(* Split back checkers: one on 24, one on 23 instead of both on 24 *)
val split_back_checkers : unit -> board

(* Slotted 11-point: move one mid-point checker to 11-point *)
val slotted_11_point : unit -> board

(* Split mid-point: checkers on 12 and 14 instead of all on 13 *)
val split_mid_point : unit -> board

(* Advanced anchor: move back checkers forward to 20-point *)
val advanced_anchor : unit -> board

(* Random perturbation: slightly randomize checker positions *)
val random_perturbation : rng_state -> board * rng_state


(* ============================================================================
   POSITION GENERATOR
   ============================================================================ *)

(* Generate a diverse set of starting positions for training *)
val training_positions : int -> rng_state -> board list * rng_state
(* Returns: list of diverse starting positions *)

(* Position distribution for training phases *)
type training_phase =
  | EarlyPhase   (* First 1000 games: more standard positions *)
  | MidPhase     (* 1000-5000 games: mix of standard and variants *)
  | LatePhase    (* 5000+ games: all variants equally *)

(* Sample starting position based on training phase *)
val sample_start_position : training_phase -> rng_state -> board * rng_state


(* ============================================================================
   MIDGAME AND ENDGAME POSITIONS
   ============================================================================ *)

(* Generate random contact position (checkers still in contact) *)
val random_contact_position : rng_state -> board * rng_state

(* Generate random race position (past contact, pure running game) *)
val random_race_position : rng_state -> board * rng_state

(* Generate random bearing-off position *)
val random_bearoff_position : rng_state -> board * rng_state

(* Generate random backgame position (checkers on opponent's home board) *)
val random_backgame_position : rng_state -> board * rng_state


(* ============================================================================
   VALIDATION
   ============================================================================ *)

(* Check if a board position is valid (15 checkers each, legal placement) *)
val is_valid_position : board -> bool

(* Repair an invalid position (adjust checker counts to be legal) *)
val repair_position : board -> board
