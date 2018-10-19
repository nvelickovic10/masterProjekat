package raf.nvelickovic10.masterProjekat.util.logger;

public enum Level {
	ERROR(1), WARNING(2), INFO(3), DEBUG(4);
	
	private final int value;
	
	private Level(int value) {
		this.value = value;
	}
	
	public int getValue() {
		return this.value;
	}
}
