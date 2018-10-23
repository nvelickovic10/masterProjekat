package raf.nvelickovic10.masterProjekat.util;

import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import raf.nvelickovic10.masterProjekat.util.logger.Level;

/**
 * Application config
 */
public class AppConfig {
	// Image config
	public static final int height = 100;
	public static final int width = 100;
	public static final int channels = 3;

	// Net config
	public static final int batchSize = 20;
	public static final int epochs = 100;
	public static final long seed = 42;

	// Data config
	public static final String imagesConcreteDirectory = "mias";
	public static final String imagesExtension = ".pgm";
	public static final double splitTrainTest = 0.8;
	public static final String imagesBasePath = FilenameUtils.concat(System.getProperty("user.dir"),
			"src/main/resources/images/");
	public static final String imagesConcretePath = imagesBasePath + imagesConcreteDirectory;
	public static final String modelsBasePath = FilenameUtils.concat(System.getProperty("user.dir"),
			"src/main/resources/models/");
	public static final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

	// App config
	public static final Level logLevel = Level.DEBUG;
	public static final boolean saveModel = true;
	public static final boolean startUIServer = true;
	public static final boolean trainWithTransforms = false;
	public static final Random rnd = new Random(seed);
}
