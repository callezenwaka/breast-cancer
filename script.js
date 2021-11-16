let trainButton, testButton, model, file;

const inputElement = document.querySelector("#file");
inputElement.addEventListener("change", handleFile, false);


// Test Cases:
// Malignant
// const testVal = tf.tensor2d([2.055083376,-0.9755653561,2.037434939,2.04649921,0.2674555846,0.9553258432,1.360279816,1.992216701,0.5978154387,-0.1230335394,-0.2151797805,0.1830914099,-0.2054748597,-0.0257061767,-0.9073612571,0.09643919749,0.09324323437,0.3027902506,-0.4611027076,0.251918026,1.317025816,-0.6069566048,1.332131208,1.116191371,-0.5866686711,0.210134298,0.6765119868,1.25905889,-0.2767866699,0.1780661916], [1, 30]);
// Malignant
// const testVal = tf.tensor2d([0.3815675768,-1.74824194,0.4398250588,0.2322171451,2.124655378,1.034636699,1.488953499,1.582412838,0.5722379129,1.177740412,0.06469273885,-1.018869576,-0.009946784933,0.09132907804,-0.5669326934,-0.1059561913,0.3719545727,0.2637457429,-0.318201638,0.1755797923,0.5186922936,-1.394626457,0.5311131818,0.3769692998,0.9132726375,0.6959424879,1.582147395,1.046028789,0.4955949451,0.9839401465], [1, 30]);
// Benign
// const testVal = tf.tensor2d([-1.807594969,1.525403399,-1.805063876,-1.327713251,-1.119640511,-1.08137694,-1.12569574,-1.262870941,0.2214604169,1.494859117,-0.08738893204,0.4657101757,-0.1637163513,-0.4435094517,1.991386203,-0.8788321422,-1.012912504,-1.977069482,1.013098924,1.212859108,-1.46956758,0.8830521995,-1.473796513,-1.080988319,-0.3034939108,-1.10148243,-1.341360042,-1.754014329,0.2444914571,0.8928999926], [1, 30]);
// Benign
// const testVal = tf.tensor2d([-0.02336094999,0.5047963229,-0.08351570354,-0.1409251375,-0.4144849646,-0.6715380913,-0.9375748157,-0.7711063063,-1.068377668,-0.63394701,-0.505613527,0.5414446162,-0.5690195214,-0.3820554559,0.1357512914,-0.6422467978,-0.7779067085,-0.2726918413,-0.9643098926,-0.3012464614,-0.1895728317,0.7780295525,-0.2589684361,-0.2828674914,-0.1663311363,-0.6575204367,-1.028900783,-0.5336123938,-1.077775011,-0.4114469036], [1, 30]);
// showData = document.querySelector('#showData');
// showData.addEventListener("click", function() {testModel();});
// trainButton = document.querySelector('#trainButton');
testButton = document.querySelector('#testButton');
// trainButton.addEventListener("click", async function () {
//     // train(model, data);
//     const model = await getModel();
//     tfvis.show.modelSummary({ name: 'Model Architecture' }, model);
//     await trainModel(model);
//     alert("Training is done, try classifying your data!");
//   });
testButton.addEventListener("click", function() {testModel();});
	
async function getModel() {
  let {numOfFeatures} = await data();
  // In the space below create a neural network that predicts 1 if the diagnosis is malignant
  // and 0 if the diagnosis is benign. Your neural network should only use dense
  // layers and the output layer should only have a single output unit with a
  // sigmoid activation function. You are free to use as many hidden layers and
  // neurons as you like.  
  // HINT: Make sure your input layer has the correct input shape. We also suggest
  // using ReLu activation functions where applicable. For this dataset only a few
  // hidden layers should be enough to get a high accuracy.
	model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [numOfFeatures], activation: "sigmoid", units: 31 }))
  model.add(tf.layers.dense({activation: "sigmoid", units: 15 }))
  model.add(tf.layers.dense({activation: "sigmoid", units: 7 }))
  model.add(tf.layers.dense({activation: "sigmoid", units: 1}));

	// Compile the model using the binaryCrossentropy loss, 
  // the rmsprop optimizer, and accuracy for your metrics. 
  model.compile({loss: "binaryCrossentropy", optimizer: tf.train.rmsprop(0.05), metrics: ['accuracy']});

	return model;
}

async function data() {
  const trainingUrl = 'wdbc-train.csv';
  const testingUrl = 'wdbc-test.csv';
  // Take a look at the 'wdbc-train.csv' file and specify the column
  // that should be treated as the label in the space below.
  // HINT: Remember that you are trying to build a classifier that 
  // can predict from the data whether the diagnosis is malignant or benign.
  const trainingData = tf.data.csv(trainingUrl, {
    columnConfigs:{
      diagnosis:{
        isLabel: true
      }
    } 
  });

  // Take a look at the 'wdbc-test.csv' file and specify the column
  // that should be treated as the label in the space below..
  // HINT: Remember that you are trying to build a classifier that 
  // can predict from the data whether the diagnosis is malignant or benign.
  const testingData = tf.data.csv(testingUrl, {
    columnConfigs:{
      diagnosis:{
        isLabel: true
      }
    }
  });

  // Specify the number of features in the space below.
  // HINT: You can get the number of features from the number of columns
  // and the number of labels in the training data. 
  let numOfFeatures = (await trainingData.columnNames()).length - 1;

  return {numOfFeatures, trainingData, testingData};
}

async function trainModel(model) {
  // console.log(data())
  // let inputFeatures = await inputFeatures();
  const {trainingData, testingData} = await data();
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
	const container = { name: 'Model Training', styles: { height: '640px' } };
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  // Convert the training data into arrays in the space below.
  // Note: In this case, the labels are integers, not strings.
  // Therefore, there is no need to convert string labels into
  // a one-hot encoded array of label values like we did in the
  // Iris dataset example. 
  const convertedTrainingData = trainingData.map(({xs,ys}) => {
    return{xs:Object.values(xs), ys: Object.values(ys)};
  }).batch(10);
        
  // Convert the testing data into arrays in the space below.
  // Note: In this case, the labels are integers, not strings.
  // Therefore, there is no need to convert string labels into
  // a one-hot encoded array of label values like we did in the
  // Iris dataset example. 
  const convertedTestingData = testingData.map(({xs,ys}) => {
    return{xs:Object.values(xs), ys: Object.values(ys)};
  }).batch(10);

  // fit model to dataset
  await model
  .fitDataset(convertedTrainingData, {
    epochs: 20,
    validationData: convertedTestingData,
    callbacks: fitCallbacks
  });

  // save model
  await model.save('localstorage://breast_cancer');
}

async function testModel() {
  const testVal = await getFile();
  console.log('prediction')
  console.log(testVal)
  const model = await tf.loadLayersModel('localstorage://breast_cancer');
  const prediction = model.predict(testVal);
  let pred = await prediction.data().then(prediction => prediction);
  console.log(pred[0])
  pred = pred[0] > 0.5 ? 1 : 0;
  
  const classNames = ["Benign", "Malignant"];
  
  alert(`Prediction: ${prediction}\nClass: ${classNames[pred]}`)
}

document.addEventListener('DOMContentLoaded', async function () {
  testButton.disabled = true;
  const model = await getModel();
  tfvis.show.modelSummary({ name: 'Model Architecture' }, model);
  await trainModel(model);
  alert("Training is done, try classifying your data!");
  testButton.disabled = false;
});

async function handleFile() {
  file = this.files[0]; /* now you can work with the file list */
  console.log(file)
}

function readUploadedFileAsText(inputFile) {
  // TODO: Read uploaded exam
  const reader = new FileReader();
  return new Promise((resolve, reject) => {
    reader.onerror = () => {
      reader.abort();
      reject(new DOMException("Problem parsing input file."));
    };
    reader.onload = () => {
      resolve(reader.result);
    };
    reader.readAsText(inputFile);
  });
};

async function getFile() {
  // TODO: Get uploaded test data
  var regex = /^([a-zA-Z0-9\s_\\.\-:])+(.csv)$/;
  if (!regex.test(file.name.toLowerCase())) return;
  if (typeof file == "undefined") return;
  try {
    const fileContents = await readUploadedFileAsText(file);
    const rows = fileContents.split("\n");
    const testTitle = rows[0];
    let testVal = rows[1].split(",");
    console.log(testVal)
    testVal.shift();
    testVal = testVal.map(Number)
    console.log(testTitle)
    console.log(testVal.length)
    testVal = tf.tensor2d(testVal, [1, testVal.length])
    return testVal;
  } catch (error) {
    console.log(error);
    return;
  }
};