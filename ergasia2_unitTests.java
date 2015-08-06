/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ergasia2pkg;

import java.util.Map;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.meta.HOMER;
import mulan.classifier.lazy.IBLR_ML;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.trees.J48;

/**
 *
 * @author nickfortress
 */ 
public class ergasia2_unitTests {
    
    public static void baseClassUnitTest() throws InvalidDataFormatException, Exception {
        ML_RUS underTester = new ML_RUS(0.1f);
        System.out.println("Unit tests for base class. Tests performed on the bibtex dataset.");
        MultiLabelInstances mli = new MultiLabelInstances("bibtex.arff", "bibtex.xml");
        System.out.println("Testing label counts: ");
        underTester.labelCount(mli);
        for (Map.Entry<String, Double> entry : underTester.labelCounts.entrySet()) {
            System.out.println("Label: " + entry.getKey() + ", Count: " + entry.getValue());
            System.out.println("Imbalance Ratio: " + underTester.imbalanceRatioPerLabel(mli, entry.getKey()));
        }
        
        System.out.println("Testing Mean Imbalance Ratio Calculation: ");
        System.out.println("Bibtex Mean Imbalance Ratio: " + underTester.meanImbalanceRatio(mli));
        System.out.println("Testing CVIR Calculation: ");
        System.out.println("Bibtex CVIR: " + underTester.calculateCVIR(mli));
    }
    
    public static void main(String args[]) throws Exception {
        baseClassUnitTest();
    }
}
