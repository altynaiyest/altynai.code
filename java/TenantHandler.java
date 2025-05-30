package org.example.eeend.server;

import org.example.eeend.client.Tenant;

import java.io.*;
import java.net.*;
import java.sql.*;

public class TenantHandler implements Runnable {
    private Socket socket;

    public TenantHandler(Socket socket) {
        this.socket = socket;
    }

    @Override
    public void run() {
        try (ObjectInputStream inputStream = new ObjectInputStream(socket.getInputStream())) {
            Tenant tenant = (Tenant) inputStream.readObject();  // получаем объект арендатора

            // Сохраняем данные арендатора в базу данных
            DatabaseHelper.saveTenantToDatabase(tenant);

            // Можно также сохранить данные в файл
            FileHelper.saveToFile("tenants.dat", tenant);

            System.out.println("Tenant data received and saved: " + tenant.getFullName());
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}
