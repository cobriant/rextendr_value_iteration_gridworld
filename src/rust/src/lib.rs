use extendr_api::prelude::*;

/// Do value iteration for GridWorld
/// @export
#[extendr]
fn value_iteration (reward: Vec<f64>, obstacles: Vec<i32>, end_cell: i32, wind: f64, beta: f64) -> Vec<f64> {
    let end_cell = end_cell as usize;
    let mut future_value: Vec<Vec<f64>> = vec![vec![0.0; 4]; 25];
    let mut value: Vec<f64> = vec![0.0; 25];
    let mut value_next: Vec<f64> = vec![1.0; 25];
    let mut value_action = vec![vec![0.0; 4]; 25];

    while check_convergence(value, value_next.clone()) {
        value = value_next.clone();

        let future_value_next = update_future_value(&mut future_value, &value, end_cell, &obstacles, wind);

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
    end_cell: usize,
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
    // End state is position 4: the value of leaving it is always 0.
    for i in 0..4 {
        future_value[end_cell][i] = 0.0;
    }
    future_value
}


// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod rust;
    fn value_iteration;
}
