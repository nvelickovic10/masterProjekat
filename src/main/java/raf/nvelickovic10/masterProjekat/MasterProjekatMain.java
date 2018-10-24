package raf.nvelickovic10.masterProjekat;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.datavec.api.split.InputSplit;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import raf.nvelickovic10.masterProjekat.factory.TransformFactory;
import raf.nvelickovic10.masterProjekat.net.Net;
import raf.nvelickovic10.masterProjekat.net.models.LeNetCustom2;
import raf.nvelickovic10.masterProjekat.util.AppConfig;
import raf.nvelickovic10.masterProjekat.util.DataManipulator;
import raf.nvelickovic10.masterProjekat.util.NetMonitor;
import raf.nvelickovic10.masterProjekat.util.logger.Logger;

public class MasterProjekatMain {

	private static Logger LOG = new Logger(MasterProjekatMain.class.getSimpleName());

	public void run() throws IOException {
		long startTime = System.nanoTime();
		LOG.info("Starting MasterProjekatMain...");

		// Load data to be classified
		// Data will be split in trainData = data[0] and testData = data[1]
		DataManipulator dataManipulator = new DataManipulator();
		InputSplit[] data = dataManipulator.readData();
		InputSplit trainData = data[0];
		InputSplit evaluationData = data[1];
		int numberOfLabels = dataManipulator.getNumberOfLabels();
		LOG.info("Data loaded! numberOfImages: " + dataManipulator.getNumberOfImages() + ", numberOfLabels: "
				+ numberOfLabels);
		LOG.debug("trainData: " + trainData.length() + ", evaluationData: " + evaluationData.length());

		// Get image transformations
		// Image transformations can be used to further expand the test data with
		// slightly modified images
		List<ImageTransform> transforms = TransformFactory.getTransforms();
		LOG.info("Transforms loaded!");

		// Build net
		Net net = new LeNetCustom2(numberOfLabels);
		net.build();
		LOG.info("Net built!");
		net.init();
		LOG.info("Net initialized!");
		NetMonitor.getInstance().attach(net.getModel());
		LOG.info("Net monitor attached to net!");

		LOG.info("Train model without transformations...");

		DataSetIterator dataSetIterator = dataManipulator.getDataSetIterator(trainData, null);
		net.train(dataSetIterator);

		if (AppConfig.trainWithTransforms) {
			// Train with transformations
			for (ImageTransform transform : transforms) {
				LOG.info("Train model with transformation: " + transform.getClass().toString());
				dataSetIterator = dataManipulator.getDataSetIterator(trainData, transform);
				net.train(dataSetIterator);
			}
		}

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

		NetMonitor.getInstance().stop();
		LOG.info("UI server stopped!");

		if (AppConfig.saveModel) {
			String name = net.saveModel();
			LOG.info("Model saved! " + name);
		}

		long totalTime = TimeUnit.NANOSECONDS.toMinutes(System.nanoTime() - startTime);
		LOG.info("MasterProjekatMain example finished!!! totalTime: " + totalTime + " minutes");
	}

	public static void main(String[] args) {
		try {
			new MasterProjekatMain().run();
		} catch (IOException e) {
			// catch recordReader.initialize()
			LOG.error("Application error!!!");
			e.printStackTrace();
		}
	}
}
