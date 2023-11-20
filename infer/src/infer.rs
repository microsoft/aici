use candle_core::{Device, Result, Tensor};

struct Model {
    first: Tensor,
    second: Tensor,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = image.matmul(&self.first)?;
        let x = x.relu()?;
        x.matmul(&self.second)
    }
}

fn main() -> Result<()> {
    println!("start");
    let device = Device::new_cuda(0)?;
    // let device = Device::Cpu;

    println!("start2");

    let first = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let second = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let model = Model { first, second };

    let _ = first.to_dtype(candle_core::DType::BF16);

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    println!("start3");

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
