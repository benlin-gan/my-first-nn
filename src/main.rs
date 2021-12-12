use nn::model::Model;
use nn::dataset;
fn main(){
    let k = Model::new(vec![3, 2]);
    println!("{}", k);
    dataset::read();
}
