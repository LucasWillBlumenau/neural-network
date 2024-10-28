use std::f64::consts::E;

use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize, Serialize)]
pub enum ActivationType {
    Relu,
    Sigmoid,
    Tanh,
}


#[derive(Debug, Deserialize, Serialize)]
pub struct Activation {
    pub activation_type: ActivationType,
}

impl Activation {

    pub fn function(&self, value: f64) -> f64 {
        match self.activation_type {
            ActivationType::Relu => { relu(value) }
            ActivationType::Sigmoid => { sigmoid(value) }
            ActivationType::Tanh => { tahn(value) }
        }
    }

    pub fn derivative(&self, value: f64) -> f64 {
        match self.activation_type {
            ActivationType::Relu => { relu_derivative(value) }
            ActivationType::Sigmoid => { sigmoid_derivative(value) }
            ActivationType::Tanh => { tahn_derivative(value) }
        }
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

fn tahn(value: f64) -> f64 {
    (E.powf(value) - E.powf(-value)) / E.powf(value) + E.powf(-value)
}

fn tahn_derivative(value: f64) -> f64 {
    let cosh = (E.powf(value) + E.powf(-value)) / 2f64;
    let sinh = (E.powf(value) - E.powf(-value)) / 2f64;
    let cosh_squared = cosh * cosh;
    let sinh_squared = sinh * sinh;
    (cosh_squared - sinh_squared) / cosh_squared
}
