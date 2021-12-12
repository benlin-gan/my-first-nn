struct Xorrng{
    state: u64,
}
impl Xorrng{
    fn seed(x: u64) -> Self{
	let mut out = Self{
	    state: x,
	};
	out.next();
	out
    }
}
impl Iterator for Xorrng{
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item>{
	self.state ^= self.state >> 12;
	self.state ^= self.state << 25;
	self.state ^= self.state >> 17;
	Some(self.state)
    }
}
impl Xorrng{
    fn rand_float(&mut self, low: f64, high: f64) -> f64{
	let x = (self.next().unwrap() as f64)/(u64::MAX as f64);
	x * (high - low) + low
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn list(){
	let mut k = Xorrng::seed(234234);
	for _ in 0..100{
	    println!("{}", k.rand_float(0.0, 1000.0));
	}
	//assert_eq!(k.rand_float(0.0, 1000.0), 1232.20);
    }
}
