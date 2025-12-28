(* agents.mli - Player agents for self-play and evaluation *)

open Types

(* ============================================================================
   AGENT INTERFACE
   ============================================================================ *)

(* Agent trait: something that can select moves *)
type agent = {
  name : string;
  select_move : board -> dice -> legal_moves -> move;
  (* Optional: can provide move probabilities for policy targets *)
  move_probabilities : (board -> dice -> legal_moves -> float array) option;
}


(* ============================================================================
   RANDOM AGENT
   ============================================================================ *)

(* Random agent: selects moves uniformly at random *)
val random_agent : rng_state -> agent * rng_state

(* Create random agent with seed *)
val random_agent_seeded : int -> agent


(* ============================================================================
   PIP COUNT HEURISTIC AGENT
   ============================================================================ *)

(* Pip count agent: minimizes pip count with penalties for bad positions *)
val pip_count_agent : unit -> agent

(* Configuration for pip count agent *)
type pip_count_config = {
  blot_penalty : float;        (* Penalty per exposed checker *)
  anchor_bonus : float;        (* Bonus for anchors in opponent home *)
  prime_bonus : float;         (* Bonus for consecutive made points *)
  race_bonus : float;          (* Bonus for being past contact *)
  hit_bonus : float;           (* Bonus for hitting opponent *)
}

(* Pip count agent with custom config *)
val pip_count_agent_custom : pip_count_config -> agent

(* Default pip count config *)
val default_pip_count_config : pip_count_config


(* ============================================================================
   NEURAL NETWORK AGENT
   ============================================================================ *)

(* Neural network agent: uses network to evaluate positions *)
val network_agent :
  network_params ->
  search_config ->
  encoding_config ->
  string ->  (* name *)
  agent

(* Greedy network agent: 0-ply, just evaluate immediate positions *)
val greedy_network_agent :
  network_params ->
  encoding_config ->
  string ->  (* name *)
  agent

(* 1-ply network agent with dice averaging *)
val one_ply_network_agent :
  network_params ->
  encoding_config ->
  string ->  (* name *)
  agent


(* ============================================================================
   MIXED AGENT (for curriculum learning)
   ============================================================================ *)

(* Mixed agent: randomly choose between two agents with probability *)
type mixed_agent_config = {
  agent_a : agent;
  agent_b : agent;
  prob_a : float;  (* Probability of using agent_a (0.0 to 1.0) *)
}

val mixed_agent : mixed_agent_config -> rng_state -> agent * rng_state


(* ============================================================================
   AGENT EVALUATION
   ============================================================================ *)

(* Play a single game between two agents *)
val play_game :
  agent ->     (* white player *)
  agent ->     (* black player *)
  board ->     (* starting position *)
  rng_state -> (* for dice *)
  game_record * rng_state

(* Play multiple games and collect statistics *)
type match_result = {
  white_wins : int;
  black_wins : int;
  white_gammons : int;
  black_gammons : int;
  white_backgammons : int;
  black_backgammons : int;
  total_games : int;
  average_plies : float;
}

val play_match :
  agent ->     (* white player *)
  agent ->     (* black player *)
  int ->       (* number of games *)
  rng_state ->
  match_result * game_record list * rng_state


(* ============================================================================
   TOURNAMENT
   ============================================================================ *)

(* Round-robin tournament between multiple agents *)
type tournament_result = {
  agents : agent list;
  win_matrix : int array array;  (* [i][j] = games agent i won vs agent j *)
  rankings : (string * float) list;  (* Agent name, win rate *)
}

val run_tournament :
  agent list ->
  int ->        (* games per pairing *)
  rng_state ->
  tournament_result * rng_state


(* ============================================================================
   ELO RATING
   ============================================================================ *)

(* Estimate ELO rating difference from match result *)
val estimate_elo_difference : match_result -> float

(* Expected score based on ELO difference *)
val expected_score : float -> float  (* elo_diff -> probability *)


(* ============================================================================
   BENCHMARKING
   ============================================================================ *)

(* Benchmark agent performance (moves per second) *)
type benchmark_result = {
  agent_name : string;
  total_moves : int;
  total_time_ms : float;
  moves_per_second : float;
  positions_evaluated : int option;  (* For network agents *)
}

val benchmark_agent :
  agent ->
  int ->  (* number of games *)
  rng_state ->
  benchmark_result * rng_state
