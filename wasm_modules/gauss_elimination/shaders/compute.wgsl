[[block]] 
struct Matrix 
{
    elements_values: array<f32>;
};


[[block]] 
struct MatrixShape 
{
    rows_number: u32;
    columns_number: u32;
};


[[block]] 
struct IterationNumber 
{
    number: u32;
};
      

      
[[group(0), binding(0)]] 
var<storage, read_write> a : Matrix;

[[group(0), binding(1)]] 
var<uniform> a_shape : MatrixShape;

[[group(0), binding(2)]] 
var<storage, read_write> b : Matrix;

[[group(0), binding(3)]] 
var<storage, read_write> iteration_number : IterationNumber;


fn forward_elimination(row: u32, iteration: u32)
{
    let coeff = a.elements_values[row * a_shape.columns_number + (iteration - 1u)] / 
        a.elements_values[(iteration - 1u) * a_shape.columns_number + (iteration - 1u)];

    b.elements_values[row] = b.elements_values[row] - coeff * b.elements_values[iteration - 1u];

    for (var column = 0u; column < a_shape.columns_number; column = column + 1u) 
    {
        let iteration_index = (iteration - 1u) * a_shape.columns_number + column;
        let index = row * a_shape.columns_number + column;

        if (column == (iteration - 1u))
        {
            a.elements_values[index] = 0.0;
        }
        else
        {
            let value = a.elements_values[index] - coeff * a.elements_values[iteration_index];
            a.elements_values[index] = value;
        }
    } 

    if (row == a_shape.rows_number - 1u) 
    {
        iteration_number.number = iteration_number.number + 1u;
    }
}


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) 
{
    // Guard against out-of-bounds work group sizes.
    if (global_id.x >= a_shape.rows_number) 
    {
        return;
    }

    let iteration = iteration_number.number;
    let row = global_id.x;

    if (iteration == 0u || row == 0u || row < iteration) 
    {
        return;
    }

    forward_elimination(row, iteration);
}
