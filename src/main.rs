use nn::model::Model;
use nn::dataset;
fn main(){
    let k = Model::new(vec![3, 2]);
    println!("{}", k);
    let mut d = dataset::read().unwrap();
    println!("{}", d[54333].1);
    println!("{:?}", dataset::translate(&d[54333]));
}
