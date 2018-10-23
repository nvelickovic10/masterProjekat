package raf.nvelickovic10.masterProjekat.util;

import static java.lang.Math.toIntExact;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

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

		BalancedPathFilter pathFilter = new BalancedPathFilter(AppConfig.rnd, this.labelMaker, numberOfImages,
				numberOfInputDataLabels, AppConfig.batchSize);

		// Split data into train and test set
		InputSplit[] returnData = fileSplit.sample(pathFilter, AppConfig.splitTrainTest, 1 - AppConfig.splitTrainTest);

		LOG.debug("Finished reading data! trainData: " + returnData[0] + ", testData: " + returnData[1]);
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
		//dataSetIterator.setPreProcessor(AppConfig.scaler);
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
