use crate::layer::Layer;


pub struct InputLayer {
    pub values: Vec<f32>
}

impl Layer for InputLayer {
    
    fn get_holded_values(&self) -> impl Iterator<Item = f32> {
        self.values.iter().map(|value| *value)
    }
    
}
