package raf.nvelickovic10.masterProjekat.util;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import raf.nvelickovic10.masterProjekat.util.logger.Logger;

public class UIServerMonitor {

	private static final Logger LOG = new Logger(UIServerMonitor.class.getSimpleName());

	private static UIServerMonitor instance = null;
	private final StatsStorage statsStorage = new InMemoryStatsStorage();
	private final UIServer uiServer = UIServer.getInstance();

	private UIServerMonitor() {
		uiServer.attach(statsStorage);
	}

	public final void attach(Model model) {
//		model.setListeners(new ScoreIterationListener(listenerFreq));
		model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
	}

	public final void stop() {
		LOG.debug("Stopping UI server...");
		this.uiServer.stop();
		LOG.debug("UI server stopped!");
	}

	public static final UIServerMonitor getInstance() {
		if (instance == null) {
			instance = new UIServerMonitor();
		}
		return instance;
	}
}
