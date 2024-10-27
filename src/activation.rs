use std::f64::consts::E;

#[derive(Debug)]
pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

impl Activation {

    pub fn sigmoid() -> Self {
        Activation { function: sigmoid, derivative: sigmoid_derivative } 
    }

    pub fn relu() -> Self {
        Activation { function: relu, derivative: relu_derivative }
    }
}

fn sigmoid(value: f64) -> f64 {
    1f64 / (1f64 + E.powf(-value))
}

fn sigmoid_derivative(value: f64) -> f64 {
    let value = sigmoid(value);
    value * (1f64 - value)
}

fn relu(value: f64) -> f64{
    if value > 0f64 {
        value
    } else {
        0f64
    }
}

fn relu_derivative(value: f64) -> f64 {
    if value > 0f64 {
        1f64
    } else {
        0f64
    } 
}
