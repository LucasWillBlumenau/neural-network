pub trait Layer {
    fn get_holded_values(&self) -> impl Iterator<Item = f64>;
}
