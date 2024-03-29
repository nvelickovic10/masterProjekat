package raf.nvelickovic10.masterProjekat.util;

import static java.lang.Math.toIntExact;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CollectScoresIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import raf.nvelickovic10.masterProjekat.net.Net;
import raf.nvelickovic10.masterProjekat.util.logger.Logger;

/**
 * Data manipulation class
 */
public class DataManipulator {

	private static final Logger LOG = new Logger(DataManipulator.class.getSimpleName());

	private int numberOfInputDataLabels, numberOfImages;
	private final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
	private final ImageRecordReader recordReader = new ImageRecordReader(AppConfig.height, AppConfig.width,
			AppConfig.channels, labelMaker);

	/**
	 * Read images from the resources folder<br />
	 * 
	 * @return InputSplit[] returnData, trainData[0], testData[1]
	 */
	public InputSplit[] readData() {
		LOG.debug("Reading data...");

		// Load images folder
		File mainPath = new File(AppConfig.imagesConcretePath);
		FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, AppConfig.rnd);

		numberOfImages = toIntExact(fileSplit.length());
		numberOfInputDataLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;

		PathFilter pathFilter = new RandomPathFilter(AppConfig.rnd, NativeImageLoader.ALLOWED_FORMATS);

		if (AppConfig.useBalancedData) {
			pathFilter = new BalancedPathFilter(AppConfig.rnd, this.labelMaker, numberOfImages, numberOfInputDataLabels,
					AppConfig.batchSize);
		}

		// Split data into train and test set
		InputSplit[] returnData = fileSplit.sample(pathFilter, AppConfig.splitTrain, 1 - AppConfig.splitTrain);
		returnData[0] = fileSplit.sample(pathFilter, 1, 0)[0];

		LOG.debug("Finished reading data[" + returnData.length + "]! trainData: " + returnData[0]);
		if (returnData.length == 2) {
			LOG.debug("evaluationData: " + returnData[1]);
		}
		return returnData;
	}

	/**
	 * Save model in the src/main/resources/models/{name}<br />
	 * 
	 * @param network - The network model
	 * @param prefix
	 * @return String modelName, the path to the save file
	 */
	public String saveModel(Model network, String prefix) {
		LOG.debug("Saving model...");
		String modelName = AppConfig.modelsBasePath + prefix + "-model.bin";
		try {
			ModelSerializer.writeModel(network, modelName, true);
		} catch (IOException e) {
			LOG.error("Model save failed!!!");
			e.printStackTrace();
		}
		LOG.debug("Finished saving model! modelName: " + modelName);
		return modelName;
	}

	/**
	 * Read model from the src/main/resources/models/{name}<br />
	 * 
	 * @param network - The network model
	 * @param prefix
	 * @return String modelName, the path to the save file
	 */
	public MultiLayerNetwork readModel(String prefix) {
		LOG.debug("Loading model...");
		String modelName = AppConfig.modelsBasePath + prefix + "-model.bin";
		try {
			MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(modelName));
			LOG.debug("Finished loading model! modelName: " + modelName);
			LOG.debug("model: " + model);
			return model;
		} catch (IOException e) {
			LOG.error("Model save failed!!!");
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Save scores in the src/main/resources/models/{name}<br />
	 * 
	 * @param collectScoresIterationListener - The scores iteration listener
	 */
	public void saveScores(CollectScoresIterationListener collectScoresIterationListener) {
		LOG.debug("Saving scores...");
		File file = new File(AppConfig.modelsBasePath + Net.LOG.getClassName() + "-scores.csv");
		try {
			collectScoresIterationListener.exportScores(file, ",");
			LOG.debug("Scores saved!");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Creates a DataSetIterator object
	 * 
	 * @param data      - data
	 * @param transform - ImageTransform to be applied
	 * @return DataSetIterator dataSetIterator
	 */
	public DataSetIterator getDataSetIterator(InputSplit data, ImageTransform transform) throws IOException {
		LOG.debug("Creating dataSetIterator...");
		recordReader.initialize(data, transform);
		DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, AppConfig.batchSize, 1,
				numberOfInputDataLabels);
		AppConfig.scaler.fit(dataSetIterator);
		if (AppConfig.useNormalize) {
			dataSetIterator.setPreProcessor(AppConfig.scaler);
		}
		LOG.debug("dataSetIterator created!");
		return dataSetIterator;
	}

	public int getNumberOfImages() {
		return this.numberOfImages;
	}

	public int getNumberOfLabels() {
		return this.numberOfInputDataLabels;
	}

	public List<String> getAllLabelsFromRecordReader() {
		return recordReader.getLabels();
	}
}
