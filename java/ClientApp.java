package org.example.eeend.client;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import java.io.*;
import java.net.*;
import java.util.*;

public class ClientApp extends Application {
    private TextField fullNameField = new TextField();
    private TextField passportField = new TextField();
    private TextField phoneField = new TextField();
    private TextField emailField = new TextField();
    private TextField incomeField = new TextField();
    private CheckBox hasPetsBox = new CheckBox("Has Pets");
    private TextField rentalDurationField = new TextField();

    @Override
    public void start(Stage stage) {
        VBox root = new VBox();
        root.getChildren().addAll(
                new Label("Full Name"), fullNameField,
                new Label("Passport Number"), passportField,
                new Label("Phone Number"), phoneField,
                new Label("Email"), emailField,
                new Label("Income"), incomeField,
                hasPetsBox,
                new Label("Rental Duration (Months)"), rentalDurationField,
                createSubmitButton()
        );

        Scene scene = new Scene(root, 300, 400);
        stage.setScene(scene);
        stage.setTitle("Tenant Form");
        stage.show();
    }

    private Button createSubmitButton() {
        Button submitButton = new Button("Submit");
        submitButton.setOnAction(event -> sendTenantData());
        return submitButton;
    }

    private void sendTenantData() {
        Tenant tenant = new Tenant(
                fullNameField.getText(),
                passportField.getText(),
                phoneField.getText(),
                emailField.getText(),
                Double.parseDouble(incomeField.getText()),
                hasPetsBox.isSelected(),
                Integer.parseInt(rentalDurationField.getText())
        );

        try (Socket socket = new Socket("localhost", 12345);
             ObjectOutputStream outputStream = new ObjectOutputStream(socket.getOutputStream())) {
            outputStream.writeObject(tenant);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        launch();
    }
}
