use extendr_api::prelude::*;

/// Do value iteration for GridWorld
/// @export
#[extendr]
fn value_iteration (reward: Vec<f64>, end_cell: i32) {
    let end_cell = end_cell as usize;
    let mut future_value: Vec<Vec<f64>> = vec![vec![0.0; 4]; 25];
    let mut value: Vec<f64> = vec![0.0; 25];
    let mut value_next: Vec<f64> = vec![1.0; 25];
    let mut value_action = vec![vec![0.0; 4]; 25];

    while check_convergence(value, value_next.clone()) {
        value = value_next.clone();

        let future_value_next = update_future_value(&mut future_value, &value, end_cell);

        for i in 0..25 {
            for j in 0..4 {
                // value_action[25x4] = reward[25x1] + 0.95 * future_value[25x4]
                value_action[i][j] = reward[i] + 0.95 * future_value_next[i][j];
            }
        }

        // value_next are row maxes of value_action:
        for i in 0..25 {
            value_next[i] = value_action[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        }
    }
    println!("{:?}", value_next);
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

fn moving (pos: usize, action: i32) -> usize {
    if hits_boundary(pos, action) {
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

fn update_future_value<'a, 'b> (
    future_value: &'a mut Vec<Vec<f64>>,
    value: &'b Vec<f64>,
    end_cell: usize
) -> &'a mut Vec<Vec<f64>> {
    for i in 0..future_value.len() {
        for j in 0..future_value[i].len() {
            let mut weights: [f64; 4] = [0.1; 4];
            weights[j] = 0.7;
            let move_vec = vec![
                moving(i, 1),
                moving(i, 2),
                moving(i, 3),
                moving(i, 4)
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
