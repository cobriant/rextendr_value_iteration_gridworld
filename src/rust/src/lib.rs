use extendr_api::prelude::*;
use std::vec;
use rand::seq::SliceRandom;
use rand::Rng;

/// Do value iteration for GridWorld
/// @export
#[extendr]
fn value_iteration (reward: Vec<f64>, obstacles: Vec<i32>, wind: f64, beta: f64) -> Vec<f64> {
    let mut future_value: Vec<Vec<f64>> = vec![vec![0.0; 4]; 25];
    let mut value: Vec<f64> = vec![0.0; 25];
    let mut value_next: Vec<f64> = vec![1.0; 25];
    let mut value_action = vec![vec![0.0; 4]; 25];

    while check_convergence(value, value_next.clone()) {
        value = value_next.clone();

        let future_value_next = update_future_value(&mut future_value, &value, &obstacles, wind);

        for i in 0..25 {
            for j in 0..4 {
                // value_action[25x4] = reward[25x1] + beta * future_value[25x4]
                value_action[i][j] = reward[i] + beta * future_value_next[i][j];
            }
        }

        // value_next are row maxes of value_action:
        for i in 0..25 {
            value_next[i] = value_action[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        }
    }

    let mut result: Vec<f64> = Vec::new();
    result.append(&mut value_next);
    result.append(&mut value_action.into_iter().flatten().collect());

    result
}


fn check_convergence (value: Vec<f64>, value_next: Vec<f64>) -> bool {
    let diff = value
    .iter()
    .zip(value_next.iter())
    .map(|(a, b)| (a - b).abs())
    .collect::<Vec<f64>>()
    .into_iter()
    .reduce(f64::max)
    .unwrap();
    diff > 0.01
}

fn moving (pos: usize, action: i32, obstacles: &Vec<i32>) -> usize {
    if hits_boundary(pos, action) {
        pos
    } else if hits_obstacles(pos, action, obstacles) {
        pos
    } else {
        match action {
            1 => pos - 5,
            2 => pos + 5,
            3 => pos - 1,
            4 => pos + 1,
            _ => 0,
        }
    }
}

fn hits_boundary (pos: usize, action: i32) -> bool {
    match action {
        1 => pos < 5,
        2 => pos > 19,
        3 => pos % 5 == 0,
        4 => pos % 5 == 4,
        _ => false
    }
}

fn hits_obstacles (pos: usize, action: i32, obstacles: &Vec<i32>) -> bool{
    let pos = pos as i32;
    for ob in obstacles.iter() {
        let hit = match action {
            1 => pos - ob == 5,
            2 => ob - pos == 5,
            3 => pos == ob + 1,
            4 => pos == ob - 1,
            _ => false
        };
        if hit {return true;}
    }
    return false;
}

fn update_future_value<'a, 'b> (
    future_value: &'a mut Vec<Vec<f64>>,
    value: &'b Vec<f64>,
    obstacles: &Vec<i32>,
    wind: f64,
) -> &'a mut Vec<Vec<f64>> {
    for i in 0..future_value.len() {
        for j in 0..future_value[i].len() {
            let mut weights: [f64; 4] = [(1.0 - wind) / 3.0; 4];
            weights[j] = wind;
            let move_vec = vec![
                moving(i, 1, &obstacles),
                moving(i, 2, &obstacles),
                moving(i, 3, &obstacles),
                moving(i, 4, &obstacles)
            ];

            let mut value_move_vec: Vec<f64> = vec![0.0; 4];
            for k in 0..4 {
                value_move_vec[k] = value[move_vec[k] as usize];
            }
            for l in 0..4 {
                value_move_vec[l] = value_move_vec[l] * weights[l];
            }
            let value_sum: f64 = value_move_vec.iter().sum();
            future_value[i][j] = value_sum;
        }
    }
    future_value
}

#[extendr]
fn generate_trajs(policy: Vec<i32>, obstacles: Vec<i32>, wind: f64) -> Vec<i32> {
    let num_trajectories = 50;
    let trajectory_length = 10;
    let mut rng = rand::thread_rng();
    let mut trajectories: Vec<Vec<i32>> = vec![vec![0; trajectory_length]; num_trajectories];

    for trajectory in trajectories.iter_mut() {
        let mut pos = loop {
            let random_pos = rng.gen_range(1..=25);
            if !obstacles.contains(&(random_pos as i32)) {
                break random_pos as i32;
            }
        };
        for t in 0..trajectory_length {
            let action = policy[(pos - 1) as usize];
            let intended_move = match action {
                1 => vec![1],
                2 => vec![2],
                3 => vec![3],
                4 => vec![4],
                12 => vec![1, 2],
                13 => vec![1, 3],
                14 => vec![1, 4],
                23 => vec![2, 3],
                24 => vec![2, 4],
                34 => vec![3, 4],
                6 => vec![1, 2, 3],
                7 => vec![1, 2, 4],
                8 => vec![1, 3, 4],
                9 => vec![2, 3, 4],
                10 => vec![1, 2, 3, 4],
                _ => vec![],
            };

            let move_prob: f64 = rng.gen();
            let actual_move = if move_prob < wind {
                *intended_move.choose(&mut rng).unwrap()
            } else {
                let mut available_moves = vec![1, 2, 3, 4];
                available_moves.retain(|m| !intended_move.contains(m));
                *available_moves.choose(&mut rng).unwrap()
            };

            let next_pos = moving(pos as usize, actual_move, &obstacles);
            if next_pos != 0 {
                pos = next_pos as i32;
            }
            trajectory[t] = pos;
        }
    }

    trajectories.into_iter().flatten().collect()
}


// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod rust;
    fn value_iteration;
    fn generate_trajs;
}
