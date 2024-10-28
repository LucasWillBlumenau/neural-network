use crate::dense_layer::DenseLayer;

pub enum Layer<'a> {
    DenseLayer(&'a DenseLayer),
    InputLayer(&'a[f64]),
}


impl Layer<'_> {
    
    pub fn get_values(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            Layer::DenseLayer(layer) => Box::new(layer.neurons.iter().map(|neuron| neuron.holded)),
            Layer::InputLayer(layer) => Box::new(layer.iter().copied()),
        }
    }

}