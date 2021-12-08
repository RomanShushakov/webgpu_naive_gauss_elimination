import init, { naive_gauss_elimination } from "../wasm/gauss_elimination/gauss_elimination.js";

await init();


// const a_rows_number = 3;
// const a_columns_number = 3;

// const a_elements_values = new Float32Array([
//     3.0, -0.1, -0.2,
//     0.1, 7.0, -0.3,
//     0.3, -0.2, 10.0,
// ]);


// const b_elements_values = new Float32Array([
//     7.85,
//     -19.3,
//     71.4,
// ]);


const a_rows_number = 4;
const a_columns_number = 4;

const a_elements_values = [
    5.0, -4.0, 1.0, 0.0,
    -4.0, 6.0, -4.0, 1.0,
    1.0, -4.0, 6.0, -4.0,
    0.0, 1.0, -4.0, 5.0,
];


const b_elements_values = [
   0.0, 
   1.0, 
   0.0,
   0.0,
];


const button = document.querySelector(".click");
button.addEventListener("click", async () => {
    naive_gauss_elimination(a_rows_number, a_columns_number, a_elements_values, b_elements_values)
        .then((result) => console.log(new Float32Array(result)));
    console.log("click");
});
