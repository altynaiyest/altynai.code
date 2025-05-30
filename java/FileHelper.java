package org.example.eeend.server;

import org.example.eeend.client.Tenant;

import java.io.*;
import java.util.*;

public class FileHelper {
    public static void saveToFile(String fileName, Tenant tenant) {
        List<Tenant> tenants = loadFromFile(fileName);
        tenants.add(tenant);
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName))) {
            oos.writeObject(tenants);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unchecked")
    public static List<Tenant> loadFromFile(String fileName) {
        List<Tenant> tenants = new ArrayList<>();
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName))) {
            tenants = (List<Tenant>) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return tenants;
    }
}


