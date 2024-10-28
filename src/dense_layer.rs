use serde::{Deserialize, Serialize};

use crate::neuron::Neuron;


#[derive(Debug, Serialize, Deserialize)]
pub struct DenseLayer {
    pub neurons: Vec<Neuron>
}


impl DenseLayer {
    
    pub fn new(size: u16, sibling_size: u16) -> Self {
        let mut neurons = vec![];
        for _ in 0..size {
            neurons.push(Neuron::new(sibling_size));
        }
        
        DenseLayer { neurons }
    }
    
}
