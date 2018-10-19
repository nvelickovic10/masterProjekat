package raf.nvelickovic10.masterProjekat.factory;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;

import raf.nvelickovic10.masterProjekat.util.AppConfig;
import raf.nvelickovic10.masterProjekat.util.logger.Logger;

public abstract class TransformFactory {
	
	private static final Logger LOG = new Logger(TransformFactory.class.getSimpleName());

	private static final ImageTransform flipTransform1 = new FlipImageTransform(AppConfig.rnd);
	private static final ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
	private static final ImageTransform warpTransform = new WarpImageTransform(AppConfig.rnd, 42);
//    ImageTransform colorTransform = new ColorConversionTransform(new Random(AppConfig.seed), COLOR_BGR2YCrCb);

	private static final List<ImageTransform> transforms = Arrays
			.asList(new ImageTransform[] { flipTransform1, warpTransform, flipTransform2 });

	public static List<ImageTransform> getTransforms() {
		LOG.debug("Transforms loaded! transforms: " + transforms.toString());
		return transforms;
	}
}
