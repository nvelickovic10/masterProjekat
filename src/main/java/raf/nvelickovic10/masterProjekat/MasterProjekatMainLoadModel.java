package raf.nvelickovic10.masterProjekat;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.datavec.api.split.InputSplit;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import raf.nvelickovic10.masterProjekat.net.Net;
import raf.nvelickovic10.masterProjekat.net.models.LeNetCustom2;
import raf.nvelickovic10.masterProjekat.util.DataManipulator;
import raf.nvelickovic10.masterProjekat.util.logger.Logger;

public class MasterProjekatMainLoadModel {

	private static Logger LOG = new Logger(MasterProjekatMainLoadModel.class.getSimpleName());

	public void run() throws IOException {
		long startTime = System.nanoTime();
		LOG.info("Starting example...");

		// Load data to be classified
		// Data will be split in trainData = data[0] and testData = data[1]
		DataManipulator dataManipulator = new DataManipulator();
		InputSplit[] data = dataManipulator.readData();
		InputSplit evaluationData = data[0];
		int numberOfLabels = dataManipulator.getNumberOfLabels();
		LOG.info("Data loaded! numberOfImages: " + dataManipulator.getNumberOfImages() + ", numberOfLabels: "
				+ numberOfLabels);
		LOG.debug("evaluationData: " + evaluationData.length());

		// Load net
		MultiLayerNetwork model = dataManipulator.readModel(LeNetCustom2.class.getSimpleName());
		Net net = new LeNetCustom2(model);
		LOG.info("Net loaded!");

		// Evaluate model with test data
		DataSetIterator testDataSetIterator = dataManipulator.getDataSetIterator(evaluationData, null);
		Evaluation evaluation = net.evaluate(testDataSetIterator);
		LOG.info("Model evaluated!");
		LOG.info(evaluation.stats(true));

		// Make a single prediction
		testDataSetIterator = dataManipulator.getDataSetIterator(evaluationData, null);
		DataSet singlePredictionDataSet = testDataSetIterator.next();

		LOG.info("Make a single prediction");
		int[] prediction = net.predict(singlePredictionDataSet);

		List<String> allClassLabels = dataManipulator.getAllLabelsFromRecordReader();
		int labelIndex = singlePredictionDataSet.getLabels().argMax(1).getInt(0);
		String expectedResult = allClassLabels.get(labelIndex);
		String modelPrediction = allClassLabels.get(prediction[0]);
		LOG.info("For a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction);

		long totalTime = TimeUnit.NANOSECONDS.toMinutes(System.nanoTime() - startTime);
		LOG.info("Example finished!!! totalTime: " + totalTime + " minutes");
	}

	public static void main(String[] args) {
		try {
			new MasterProjekatMainLoadModel().run();
		} catch (IOException e) {
			// catch recordReader.initialize()
			LOG.error("Application error!!!");
			e.printStackTrace();
		}
	}
}
