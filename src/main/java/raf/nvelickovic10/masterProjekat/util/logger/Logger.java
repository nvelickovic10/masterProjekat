package raf.nvelickovic10.masterProjekat.util.logger;

import raf.nvelickovic10.masterProjekat.util.AppConfig;

public class Logger {

	private String className;

	private String sufix = ".java >>> ";

	public Logger(String className) {
		this.className = className;
	}
	
	public void error(String log) {
		if (AppConfig.logLevel.getValue() >= Level.ERROR.getValue()) {
			System.err.println("[ERROR] " + className + sufix + log);
		}
	}
	
	public void warning(String log) {
		if (AppConfig.logLevel.getValue() >= Level.WARNING.getValue()) {
			System.out.println("[WARNING] " + className + sufix + log);
		}
	}

	public void info(String log) {
		if (AppConfig.logLevel.getValue() >= Level.INFO.getValue()) {
			System.out.println("[INFO] " + className + sufix + log);
		}
	}
	
	public void debug(String log) {
		if (AppConfig.logLevel.getValue() >= Level.DEBUG.getValue()) {
			System.out.println("[DEBUG] " + className + sufix + log);
		}
	}
	
	public String getClassName() {
		return className;
	}
}
