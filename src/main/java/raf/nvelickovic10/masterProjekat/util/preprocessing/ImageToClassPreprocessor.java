package raf.nvelickovic10.masterProjekat.util.preprocessing;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import org.apache.commons.io.FileUtils;

import raf.nvelickovic10.masterProjekat.util.AppConfig;
import raf.nvelickovic10.masterProjekat.util.logger.Logger;

/**
 * Preprocessor class used to categorize images into corresponding directories
 */
public class ImageToClassPreprocessor {

	private final Logger LOG = new Logger(ImageToClassPreprocessor.class.getSimpleName());

	private final String csvFile = AppConfig.imagesConcretePath + "/kategorizacija.csv";
	private final String SEPARATOR = ",";

	private final int SNIMAK = 0, TKIVO = 1, ANOMALIJA = 2, KLASIFIKACIJA = 3, X = 4, Y = 5, R = 6;

	private final void preprocess() {
		String line;
		File file;
		try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
			br.readLine();
			while ((line = br.readLine()) != null) {
				String[] imageData = line.split(SEPARATOR);

				file = new File(AppConfig.imagesConcretePath + "/" + imageData[SNIMAK] + AppConfig.imagesExtension);
				LOG.info("Image [ime= " + imageData[SNIMAK] + ", tkivo= " + imageData[TKIVO] + ", anomalija= "
						+ imageData[ANOMALIJA] + ", klasifikacija= " + imageData[KLASIFIKACIJA] + ", datoteka= "
						+ file.getAbsolutePath() + "]");

				FileUtils.copyFile(file, new File(AppConfig.imagesBasePath + "mias1/" + imageData[ANOMALIJA] + "/"
						+ imageData[SNIMAK] + AppConfig.imagesExtension));

				if (imageData[KLASIFIKACIJA].equals("N")) {
					FileUtils.copyFile(file, new File(
							AppConfig.imagesBasePath + "mias2/N/" + imageData[SNIMAK] + AppConfig.imagesExtension));
				} else {
					FileUtils.copyFile(file, new File(
							AppConfig.imagesBasePath + "mias2/A/" + imageData[SNIMAK] + AppConfig.imagesExtension));
				}

			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		ImageToClassPreprocessor itcp = new ImageToClassPreprocessor();
		itcp.preprocess();
	}

}
