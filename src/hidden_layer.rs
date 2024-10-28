use serde::{Deserialize, Serialize};

use crate::layer::Layer;
use crate::neuron::Neuron;


#[derive(Debug, Serialize, Deserialize)]
pub struct HiddenLayer {
    pub neurons: Vec<Neuron>
}


impl HiddenLayer {
    
    pub fn new(size: u16, sibling_size: u16) -> Self {
        let mut neurons = vec![];
        for _ in 0..size {
            neurons.push(Neuron::new(sibling_size));
        }
        
        HiddenLayer { neurons }
    }
    
}


impl Layer for HiddenLayer {
    
    fn get_holded_values(&self) -> impl Iterator<Item = f64> {
        self.neurons.iter().map(|neuron| neuron.holded)
    }
}
