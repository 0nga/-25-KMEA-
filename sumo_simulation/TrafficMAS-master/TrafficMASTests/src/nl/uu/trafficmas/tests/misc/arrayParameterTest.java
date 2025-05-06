package nl.uu.trafficmas.tests.misc;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import org.junit.Test;

public class arrayParameterTest {

	@Test
	public void arrayParameter() {
		double [] array = new double[1];
		array(array);
		assertNotNull(array);
		assertEquals(array[0],5,0);
	}
	
	private void array(double[] array) {
		array[0] = 5;
	}

}
