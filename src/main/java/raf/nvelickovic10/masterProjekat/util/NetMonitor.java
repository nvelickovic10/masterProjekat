package raf.nvelickovic10.masterProjekat.util;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.listeners.CollectScoresIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import raf.nvelickovic10.masterProjekat.util.logger.Logger;

public class NetMonitor {

	private static final Logger LOG = new Logger(NetMonitor.class.getSimpleName());

	private static NetMonitor instance = null;
	private final CollectScoresIterationListener collectScoresIterationListener = new CollectScoresIterationListener(1);
	private final ScoreIterationListener scoreIterationListener = new ScoreIterationListener(1);
	private final DataManipulator dataManipulator = new DataManipulator();

	public final void attach(Model model) {
		if (AppConfig.startUIServer) {
			StatsStorage statsStorage = new InMemoryStatsStorage();
			StatsListener statsListener = new StatsListener(statsStorage);
			UIServer.getInstance().attach(statsStorage);
			model.addListeners(statsListener);
		}
		model.addListeners(collectScoresIterationListener);
		model.addListeners(scoreIterationListener);
	}

	private final void saveScore() {
		dataManipulator.saveScores(collectScoresIterationListener);
	}

	public final void stop() {
		LOG.debug("Saving scores...");
		saveScore();
		LOG.debug("Scores saved!");
		if (AppConfig.startUIServer) {
			LOG.debug("Stopping UI server...");
			UIServer.getInstance().stop();
			LOG.debug("UI server stopped!");
		}
	}

	public static final NetMonitor getInstance() {
		if (instance == null) {
			instance = new NetMonitor();
		}
		return instance;
	}
}
