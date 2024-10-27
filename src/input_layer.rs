use crate::layer::Layer;

#[derive(Debug)]
pub struct InputLayer {
    pub values: Vec<f64>
}

impl Layer for InputLayer {
    
    fn get_holded_values(&self) -> impl Iterator<Item = f64> {
        self.values.iter().map(|value| *value)
    }
    
}
