package raf.nvelickovic10.masterProjekat.net;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import raf.nvelickovic10.masterProjekat.net.models.LeNet;
import raf.nvelickovic10.masterProjekat.util.AppConfig;
import raf.nvelickovic10.masterProjekat.util.DataManipulator;
import raf.nvelickovic10.masterProjekat.util.logger.Logger;

public abstract class Net {
	public static Logger LOG;
	protected MultiLayerNetwork model;
	protected final int numberOfLabels;

	private final DataManipulator dataManipulator = new DataManipulator();
	private MultipleEpochsIterator trainIter;
	private Evaluation evaluation;

	protected Net(String className, int numberOfLabels) {
		LOG = new Logger(className);
		this.numberOfLabels = numberOfLabels;
	}

	public abstract void build();

	public final void init() {
		LOG.debug("Initializing net...");
		model.init();
		LOG.debug("Net initialized!");
	}

	public final void train(DataSetIterator dataSetIterator) {
		LOG.debug("Training model...");
		trainIter = new MultipleEpochsIterator(AppConfig.epochs, dataSetIterator);
		model.fit(trainIter);
		LOG.debug("Model trained!");
	}

	public final Evaluation evaluate(DataSetIterator dataSetIterator) {
		LOG.debug("Evaluating model...");
		evaluation = model.evaluate(dataSetIterator);
		LOG.debug(evaluation.stats(true));
		LOG.debug("Model evaluated!");
		return evaluation;
	}

	public final int[] predict(DataSet testDataSet) {
		LOG.debug("Making prediction...");
		int[] predictedClasses = model.predict(testDataSet.getFeatures());
		LOG.debug("predictions: " + predictedClasses);
		LOG.debug("Prediction done!");
		return predictedClasses;
	}

	public final String saveModel() {
		LOG.debug("Saving model...");
		String name = dataManipulator.saveModel(this.model, Net.LOG.getClassName());
		LOG.debug("Model saved: " + name);
		return name;
	}
	
	public final Model getModel() {
		return this.model;
	}
}
