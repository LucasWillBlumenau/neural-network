#[derive(Debug)]
pub struct Activation {
    pub function: fn(f32) -> f32,
    pub derivative: fn(f32) -> f32,
}
