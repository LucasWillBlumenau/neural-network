use crate::activation::Activation;

pub struct NetworkConfig {
    pub activation: Activation,
    pub layers_quantity: u16,
    pub layers_size: u16,
    pub input_size: u16,
    pub output_size: u16,
    pub learning_rate: f64,
}

