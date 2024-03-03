import React, { useState } from "react";
import axios from "axios";

const Home = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
  };

  const handlePredict = async () => {
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData
      );
      setResult(response.data.result);
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <>
      <div className="flex flex-col justify-center items-center h-screen">
        <div className="mb-4">
          <h1 className="text-3xl font-bold">
            Brain Tumor Detection Using Deep Learning
          </h1>
        </div>

        <div className="mb-4">
          <input
            type="file"
            onChange={handleFileChange}
            className="p-2 border border-gray-300 rounded"
          />
          {file && <p>Selected file: {file.name}</p>}
        </div>

        <div className="mb-4">
          <button
            onClick={handlePredict}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            Predict
          </button>
        </div>

        <div>{result && <p className="text-xl"> Result: {result}</p>}</div>
      </div>
    </>
  );
};

export default Home;
