package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;

import org.junit.Test;
import org.xml.sax.SAXException;

public class XMLSimulationLengthTest {

	@Test
	public void xmlSimulationLength() throws SAXException, IOException, ParserConfigurationException {
		String dir 		= System.getProperty("user.dir")+"/tests/XML/SimulationLength/";
		String masXML 	= "MASTest.xml";
		
		DataModel dataModel 	= new DataModelXML(dir, masXML);
		int simulationLength 	= dataModel.getSimulationLength();
		assertEquals(simulationLength,1000);
	}

}
