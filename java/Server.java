package org.example.eeend.server;

import java.io.*;
import java.net.*;

public class Server {
    private static final int PORT = 12345;

    public static void main(String[] args) {
        System.out.println("Server started...");
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            while (true) {
                Socket clientSocket = serverSocket.accept();
                new Thread(new TenantHandler(clientSocket)).start();  // запуск потока для каждого клиента
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
