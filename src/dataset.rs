use std::fs::File;
use std::io::Read;
use std::io;
use std::fmt;
use std::fmt::Formatter;
use std::fmt::Error;
use crate::matrix::Matrix;
pub fn read() -> io::Result<Vec<(u8, Image)>>{
    let mut i = File::open("train-images-idx3-ubyte")?;
    let mut l = File::open("train-labels-idx1-ubyte")?;
    let mut images = Vec::new();
    let mut labels = Vec::new();
    i.read_to_end(&mut images)?;
    l.read_to_end(&mut labels)?;
    println!("{}", images.len());
    println!("{}", labels.len());
    let mut dataset = Vec::with_capacity(60000);
    for i in 0..60000{
	dataset.push((labels[i + 8], Image::new(&images[i*784+16..i*784+800])));
    }
    Ok(dataset)
}
pub fn translate(readable: &(u8, Image)) -> (Matrix, Matrix) {
    let mut one_hot = vec![0.0; 10];
    one_hot[readable.0 as usize] = 1.0;
    let mut img = readable.1.data.iter().map(|x| *x as f64).collect();
    (Matrix::as_ket(img), Matrix::as_ket(one_hot))
}
pub struct Image{
    data: Vec<u8>,
}
impl Image{
    pub fn new(data: &[u8]) -> Self{
	assert_eq!(data.len(), 784);
	Self{
	    data: data.to_vec(),
	}
    }
    fn draw(pixel: u8) -> String{
	let brightness = pixel >> 5;
	match brightness{
	    0 => "   ",
	    1 => "...",
	    2 => "..:",
	    3 => "..|",
	    4 => ".:|",
	    5 => ".||",
	    6 => ":||",
	    7 => "|||",
	    _ => "",
	}.into()
    }
}
impl fmt::Display for Image{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error>{
	for i in 0..28{
	    for j in 0..28{
		write!(f, "{}", Image::draw(self.data[i * 28 + j]))?;
	    }
	    writeln!(f)?;
	}
	Ok(())
    }
}
